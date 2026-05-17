import paho.mqtt.client as mqtt
import json
import time

# ============================================================
# CONFIGURATION THINGSBOARD
# ============================================================
THINGSBOARD_HOST = "localhost"
THINGSBOARD_PORT = 1883

# Token de VM1 (représentant pour les données statiques)
TOKEN = "wwu5jp5yayyvbmkx3wcb"

# ============================================================
# DONNÉES RÉELLES DES 8 TECHNIQUES
# ============================================================
TECHNIQUES = {
    "Q1_Dynamic":    {"accuracy": 96.02, "taille_mo": 9.34,  "latence_ms": 19.9, "compression": 1.20},
    "Q2_Static_PTQ": {"accuracy": 96.02, "taille_mo": 9.34,  "latence_ms": 15.2, "compression": 1.20},
    "Q3_QAT":        {"accuracy": 94.77, "taille_mo": 9.34,  "latence_ms": 17.2, "compression": 1.20},
    "Q4_Weight_Only":{"accuracy": 96.02, "taille_mo": 9.34,  "latence_ms": 20.1, "compression": 1.20},
    "Q5_Mixed":      {"accuracy": 96.02, "taille_mo": 9.34,  "latence_ms": 17.5, "compression": 1.20},
    "P1_Non_Struct": {"accuracy": 95.80, "taille_mo": 11.22, "latence_ms": 20.7, "compression": 1.00},
    "P2_Semi_Struct":{"accuracy": 96.36, "taille_mo": 11.22, "latence_ms": 20.0, "compression": 1.00},
    "P3_Magnitude":  {"accuracy": 96.36, "taille_mo": 11.22, "latence_ms": 18.1, "compression": 1.00},
    "Baseline":      {"accuracy": 96.00, "taille_mo": 11.22, "latence_ms": 12.0, "compression": 1.00},
}

# ============================================================
# CONNEXION MQTT
# ============================================================
def main():
    print("=" * 60)
    print("  ENVOI DONNÉES COMPARAISON TECHNIQUES")
    print("  Master Data Science — ENS Martil")
    print("=" * 60)
    print()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(TOKEN)

    connected = False

    def on_connect(c, userdata, flags, reason_code, properties):
        nonlocal connected
        if reason_code == 0:
            print("✅ Connecté à ThingsBoard")
            connected = True
        else:
            print(f"❌ Erreur connexion (code={reason_code})")

    client.on_connect = on_connect
    client.connect(THINGSBOARD_HOST, THINGSBOARD_PORT, 60)
    client.loop_start()
    time.sleep(2)

    if not connected:
        print("❌ Impossible de se connecter à ThingsBoard")
        return

    print()
    print("📊 Envoi des données des 8 techniques...")
    print()

    # Envoyer toutes les techniques en une seule télémétrie
    payload = {}
    for technique, data in TECHNIQUES.items():
        payload[f"{technique}_accuracy"]    = data["accuracy"]
        payload[f"{technique}_taille"]      = data["taille_mo"]
        payload[f"{technique}_latence"]     = data["latence_ms"]
        payload[f"{technique}_compression"] = data["compression"]

    result = client.publish(
        "v1/devices/me/telemetry",
        json.dumps(payload),
        qos=1
    )

    if result.rc == 0:
        print("✅ Données envoyées avec succès !")
        print()
        for technique, data in TECHNIQUES.items():
            print(f"  📡 {technique:<20} | Accuracy: {data['accuracy']}% | Taille: {data['taille_mo']} Mo | Latence: {data['latence_ms']}ms")
    else:
        print(f"❌ Erreur d'envoi (code={result.rc})")

    time.sleep(1)
    print()
    print("✅ Terminé ! Vérifiez le dashboard : http://localhost:9090")

    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    main()