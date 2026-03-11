"""
=============================================================
 thingsboard/mqtt_client.py
 Envoie la télémétrie des VM vers ThingsBoard via MQTT
=============================================================
AVANT DE LANCER :
  1. docker run -it -p 9090:9090 -p 1883:1883 --name thingsboard thingsboard/tb-postgres
  2. Aller sur http://localhost:9090
  3. Login : tenant@thingsboard.org / tenant
  4. Créer 3 Devices : VM1_Capteur, VM2_Gateway, VM3_EdgeServer
  5. Copier les Access Tokens de chaque device ci-dessous
=============================================================
"""
import json, time, sys, os
import psutil

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Installation de paho-mqtt...")
    os.system("pip install paho-mqtt")
    import paho.mqtt.client as mqtt

# ── CONFIGURATION ──
# Remplace par tes vrais tokens copiés depuis ThingsBoard !
TOKENS = {
    "VM1": "TON_TOKEN_VM1_ICI",
    "VM2": "TON_TOKEN_VM2_ICI",
    "VM3": "TON_TOKEN_VM3_ICI",
}
THINGSBOARD_HOST = "localhost"
THINGSBOARD_PORT = 1883


class VMMonitor:
    def __init__(self, vm_id):
        self.vm_id  = vm_id
        self.client = mqtt.Client()
        self.client.username_pw_set(TOKENS[vm_id])
        try:
            self.client.connect(THINGSBOARD_HOST, THINGSBOARD_PORT, keepalive=60)
            self.client.loop_start()
            print(f"✅ {vm_id} connectée à ThingsBoard")
        except Exception as e:
            print(f"❌ {vm_id} : connexion impossible ({e})")
            print("   → Vérifie que ThingsBoard tourne : docker ps")

    def publier(self, prediction, confiance, temps_ms, technique, patient_id=None):
        """Envoie les données d'une inférence vers ThingsBoard."""
        data = {
            "vm_id":             self.vm_id,
            "technique":         technique,
            "prediction":        prediction,
            "confidence":        round(confiance, 4),
            "inference_time_ms": round(temps_ms, 2),
            "cpu_usage_pct":     psutil.cpu_percent(interval=0.3),
            "ram_usage_mb":      round(psutil.Process().memory_info().rss / 1024**2, 1),
            "patient_id":        patient_id or f"P-{int(time.time()) % 10000:04d}"
        }
        result = self.client.publish("v1/devices/me/telemetry", json.dumps(data))
        if result.rc == 0:
            print(f"📡 [{self.vm_id}] → {prediction} ({confiance:.0%}) | {temps_ms:.0f}ms")
        else:
            print(f"❌ [{self.vm_id}] Erreur d'envoi (code={result.rc})")
        return data

    def deconnecter(self):
        self.client.loop_stop()
        self.client.disconnect()


def demo_telemetrie():
    """Démo : simule des inférences et envoie vers ThingsBoard."""
    import random

    print("\n🚀 Démo télémétrie — 10 inférences simulées\n")

    vm1 = VMMonitor("VM1")
    vm2 = VMMonitor("VM2")
    vm3 = VMMonitor("VM3")
    vms = [vm1, vm2, vm3]

    techniques = {
        "VM1": "Q1 - Quantification Dynamique",
        "VM2": "Q2 - Quantification Statique",
        "VM3": "P2 - Élagage Structuré",
    }

    for i in range(10):
        patient_id = f"P-2026-{i+1:03d}"
        print(f"\nPatient {patient_id}")
        for vm in vms:
            prediction = random.choice(["NORMAL", "PNEUMONIA"])
            confiance  = random.uniform(0.70, 0.99)
            temps_ms   = random.uniform(30, 200)
            vm.publier(
                prediction=prediction,
                confiance=confiance,
                temps_ms=temps_ms,
                technique=techniques[vm.vm_id],
                patient_id=patient_id
            )
        time.sleep(2)   # Pause 2 secondes entre chaque patient

    for vm in vms:
        vm.deconnecter()
    print("\n✅ Démo terminée !")


if __name__ == "__main__":
    print("="*55)
    print("  THINGSBOARD — Client MQTT")
    print("="*55)
    demo_telemetrie()
