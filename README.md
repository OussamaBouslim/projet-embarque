# 🩺 Edge AI — Détection de pneumonie optimisée pour l'embarqué

> Comment déployer un modèle de deep learning de **détection de pneumonie** (radios thoraciques) sur des dispositifs **IoT aux ressources contraintes** ? Ce projet compresse un MobileNetV2 via **8 techniques d'optimisation**, sélectionne la meilleure pour chaque appareil, fait **voter 3 machines** pour un diagnostic collectif, et supervise le tout en temps réel dans **ThingsBoard**.

Projet Master Data Science — ENS, Université Abdelmalek Essaâdi (Martil / Tétouan).

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-transfer%20learning-6E4AFF)
![Quantization](https://img.shields.io/badge/Edge%20AI-quantization%20%2B%20pruning-0EA5E9)
![ThingsBoard](https://img.shields.io/badge/ThingsBoard-MQTT-2E7D32)
![Docker](https://img.shields.io/badge/Docker-3%20VM%20contraintes-2496ED?logo=docker&logoColor=white)

---

## 🎯 Le problème

Un modèle de vision médicale performant est **trop lourd** pour un capteur ou une passerelle IoT (RAM/CPU limités). L'enjeu : **réduire la taille et la latence** du modèle sans perdre en précision diagnostique, puis choisir la variante adaptée à chaque type d'appareil et fiabiliser la décision.

## 🧠 L'approche

```
                    MobileNetV2 (baseline, 11.22 Mo)
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                          ▼
  5 techniques de QUANTIZATION            3 techniques de PRUNING
  (Q1 dynamic · Q2 static PTQ ·           (P1 non structuré ·
   Q3 QAT · Q4 weight-only ·               P2 structuré ·
   Q5 mixed precision)                     P3 magnitude)
         └────────────────────┬────────────────────┘
                              ▼
              Matrice 3 VM × 8 techniques = 24 déploiements testés
                              ▼
        Sélection multi-critères (RAM / vitesse / précision par VM)
                              ▼
     Intelligence collective : vote pondéré des 3 VM → diagnostic final
                              ▼
              Supervision temps réel dans ThingsBoard (MQTT)
```

---

## ✨ Points forts

- 🏗️ **Transfer learning** MobileNetV2 (ImageNet) fine-tuné sur le dataset *Chest X-Ray Pneumonia* (NORMAL vs PNEUMONIA).
- ⚡ **8 techniques d'optimisation** implémentées et benchmarkées : 5 de quantization + 3 de pruning.
- 🖥️ **3 profils d'appareils IoT** simulés par conteneurs Docker à ressources limitées :
  - **VM1 — Capteur** : 1 cœur, 500 Mo RAM (très contraint)
  - **VM2 — Gateway** : 2 cœurs, 1 Go RAM
  - **VM3 — Edge** : 2 cœurs, 2 Go RAM
- 🎯 **Sélection multi-critères** : chaque VM a ses priorités (le capteur privilégie RAM+vitesse, l'edge privilégie la précision) → score pondéré, meilleure technique choisie automatiquement.
- 🗳️ **Intelligence collective** : les 3 VM votent, décision finale par vote pondéré par la précision.
- 📊 **Supervision ThingsBoard** via MQTT : télémétrie et comparaison des techniques en direct.

---

## 📈 Résultats réels

| Technique | Accuracy | Taille | Latence | Compression |
|---|---|---|---|---|
| Baseline (MobileNetV2) | 96.0 % | 11.22 Mo | 12.0 ms | ×1.00 |
| **Q2 — Static PTQ** | 96.02 % | **9.34 Mo** | **15.2 ms** | ×1.20 |
| Q5 — Mixed precision | 96.02 % | 9.34 Mo | 17.5 ms | ×1.20 |
| **P2 / P3 — Pruning** | **96.36 %** | 11.22 Mo | ~18–20 ms | ×1.00 |

**Techniques retenues par appareil** (via `deployment/selection.py`) :

| VM | Priorités | Technique choisie |
|---|---|---|
| VM1 — Capteur | RAM 40 % · vitesse 40 % · précision 20 % | **Q2 Static PTQ** |
| VM2 — Gateway | précision 40 % · RAM 30 % · vitesse 30 % | **Q4 Weight-only** |
| VM3 — Edge | précision 60 % · vitesse 25 % · RAM 15 % | **Q5 Mixed precision** |

➡️ Modèle **~17 % plus léger** (11.22 → 9.34 Mo) tout en **maintenant ~96 % d'accuracy**, et diagnostic collectif consolidé par vote des 3 appareils.

---

## 🛠️ Stack technique

**Deep Learning** — PyTorch, torchvision, MobileNetV2, scikit-learn (métriques)
**Optimisation** — quantization (dynamic / static PTQ / QAT / weight-only / mixed), pruning (non structuré / structuré / magnitude)
**Edge / IoT** — Docker (VM contraintes), MQTT (paho-mqtt), ThingsBoard
**Analyse** — pandas / numpy, score multi-critères, matrice de résultats CSV

---

## 📁 Structure du projet

```
dataset/         preprocessing (split 70/15/15, augmentation) + README dataset
baseline/        entraînement MobileNetV2 de référence + évaluation
optimization/    les 8 modèles optimisés (Q1–Q5 quantization, P1–P3 pruning)
deployment/      test_on_vm.py (matrice 3×8) + selection.py (meilleure technique/VM)
collective/      orchestrateur.py — vote pondéré des 3 VM (intelligence collective)
thingsboard/     clients MQTT + envoi télémétrie vers les dashboards
environment/     docker-compose.yml (3 VM) + check_resources.sh
report/ results/ mesures, matrice de résultats, rapport
```

---

## 🚀 Démarrage rapide

**Prérequis :** Python 3.10, Docker, ThingsBoard (local), et le dataset Kaggle.

```bash
# 1. Dataset (Chest X-Ray Pneumonia — 5863 images)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p dataset/
python dataset/preprocessing.py          # nettoyage + split 70/15/15

# 2. Modèle de référence
python baseline/train.py

# 3. Lancer les 3 VM IoT contraintes
docker-compose -f environment/docker-compose.yml up -d
bash environment/check_resources.sh      # vérifier RAM/CPU des VM

# 4. Matrice 3 VM × 8 techniques + sélection
python deployment/test_on_vm.py
python deployment/selection.py

# 5. Diagnostic collectif + supervision ThingsBoard
python collective/orchestrateur.py
python thingsboard/send_techniques.py    # pousse la télémétrie vers les dashboards
```

> Les modèles (`*.pt`) et le dataset ne sont pas versionnés (voir `.gitignore`) — ils se régénèrent via les scripts ci-dessus.

---

## 👤 Auteur

**Oussama BOUSLIM** — Data Scientist & AI Engineer
📍 Casablanca, Maroc · ✉️ oussama.bouslim1602@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/oussama-bouslim-a79204241)

> Projet Edge AI — compression de modèle, déploiement sur dispositifs contraints, intelligence collective et supervision IoT temps réel.
