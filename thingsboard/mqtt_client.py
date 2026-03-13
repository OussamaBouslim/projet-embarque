import paho.mqtt.client as mqtt
import json
import time
import random

# ============================================================
# CONFIGURATION THINGSBOARD — TOKENS RÉELS
# ============================================================
THINGSBOARD_HOST = "localhost"
THINGSBOARD_PORT = 1883

DEVICES = {
    "VM1_Capteur": {
        "token": "jtes3x4jii0vinis0gdu",
        "modele": "Q2_Static_PTQ",
        "cores": 1,
        "ram_mb": 500
    },
    "VM2_Gateway": {
        "token": "4fpqxTyMzfIJ6qVfwPMH",
        "modele": "Q4_Weight_Only",
        "cores": 2,
        "ram_mb": 1024
    },
    "VM3_Edge": {
        "token": "ece8abdKm5NPGDX9s6PZ",
        "modele": "Q5_Mixed_Precision",
        "cores": 2,
        "ram_mb": 2048
    }
}

# Précisions des modèles (résultats réels)
ACCURACIES = {
    "VM1_Capteur": 96.02,
    "VM2_Gateway": 96.02,
    "VM3_Edge":    96.02
}

# ============================================================
# CONNEXION MQTT PAR DEVICE
# ============================================================
clients = {}

def connect_device(name, token):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(token)

    def on_connect(c, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print(f"✅ {name} connectée à ThingsBoard")
        else:
            print(f"❌ {name} erreur connexion (code={reason_code})")

    def on_publish(c, userdata, mid, reason_code, properties):
        pass

    client.on_connect = on_connect
    client.on_publish = on_publish
    client.connect(THINGSBOARD_HOST, THINGSBOARD_PORT, 60)
    client.loop_start()
    return client

# ============================================================
# SIMULATION PATIENTS
# ============================================================
def simulate_inference(vm_name):
    """Simule une inférence médicale sur une VM."""
    classes = ["NORMAL", "PNEUMONIA"]
    prediction = random.choice(classes)
    confidence = round(random.uniform(88, 99), 1)
    latency_ms = random.randint(
        80 if vm_name == "VM1_Capteur" else 120,
        160 if vm_name == "VM1_Capteur" else 220
    )
    ram_used = random.randint(
        150 if vm_name == "VM1_Capteur" else 300,
        400 if vm_name == "VM1_Capteur" else 900
    )
    return prediction, confidence, latency_ms, ram_used

def vote_collectif(results):
    """Vote majoritaire pour l'intelligence collective."""
    votes = [r["prediction"] for r in results]
    pneumonia_votes = votes.count("PNEUMONIA")
    normal_votes = votes.count("NORMAL")
    if pneumonia_votes >= 2:
        return "PNEUMONIA", pneumonia_votes
    else:
        return "NORMAL", normal_votes

# ============================================================
# ENVOI TÉLÉMÉTRIE
# ============================================================
def send_telemetry(client, vm_name, patient_id, prediction, confidence, latency_ms, ram_used, decision_collective):
    payload = {
        "patient_id": patient_id,
        "vm_name": vm_name,
        "modele": DEVICES[vm_name]["modele"],
        "prediction": prediction,
        "confidence_pct": confidence,
        "latency_ms": latency_ms,
        "ram_used_mb": ram_used,
        "ram_total_mb": DEVICES[vm_name]["ram_mb"],
        "accuracy_pct": ACCURACIES[vm_name],
        "decision_collective": decision_collective,
        "timestamp": int(time.time() * 1000)
    }

    result = client.publish(
        "v1/devices/me/telemetry",
        json.dumps(payload),
        qos=1
    )

    if result.rc == 0:
        print(f"  📡 [{vm_name}] → {prediction} ({confidence}%) | {latency_ms}ms | RAM: {ram_used}MB")
    else:
        print(f"  ❌ [{vm_name}] Erreur d'envoi (code={result.rc})")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  SUPERVISION IOT — INFÉRENCE MÉDICALE EMBARQUÉE")
    print("  Master Data Science — ENS Martil")
    print("=" * 60)
    print()

    # Connexion des 3 VMs
    for name, info in DEVICES.items():
        clients[name] = connect_device(name, info["token"])

    time.sleep(2)  # Attendre connexions
    print()

    # Simulation de 10 patients
    for i in range(1, 11):
        patient_id = f"P-2026-{i:03d}"
        print(f"Patient {patient_id}")

        results = []
        for vm_name in DEVICES:
            pred, conf, lat, ram = simulate_inference(vm_name)
            results.append({
                "vm": vm_name,
                "prediction": pred,
                "confidence": conf
            })

        # Vote collectif
        decision, nb_votes = vote_collectif(results)
        print(f"  🧠 Décision collective : {decision} ({nb_votes}/3 votes)")

        # Envoi télémétrie pour chaque VM
        for j, vm_name in enumerate(DEVICES):
            r = results[j]
            send_telemetry(
                clients[vm_name],
                vm_name,
                patient_id,
                r["prediction"],
                r["confidence"],
                *simulate_inference(vm_name)[2:],
                decision
            )

        time.sleep(1)
        print()

    # Statistiques finales
    print("=" * 60)
    print("  STATISTIQUES FINALES")
    print("=" * 60)
    for vm_name, info in DEVICES.items():
        print(f"  {vm_name} | Modèle: {info['modele']} | Accuracy: {ACCURACIES[vm_name]}%")
    print()
    print("✅ Démo terminée ! Vérifiez le dashboard : http://localhost:9090")
    print()

    # Arrêt propre
    for client in clients.values():
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()