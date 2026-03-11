"""
=============================================================
 optimization/Q1_dynamic_quant/quantize.py
 Technique : Quantification Dynamique
=============================================================
"""
import os, sys, time, json
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.train import construire_modele, evaluer, mesurer_inference
from dataset.preprocessing import creer_transforms, creer_dataloaders

DEVICE = torch.device('cpu')

# ── Charger le modèle de base ──
print("📦 Chargement du modèle baseline...")
model = construire_modele()
model.load_state_dict(torch.load("baseline/model_baseline.pt", map_location=DEVICE))
model.eval()

# ── Appliquer la quantification dynamique ──
# Les poids sont compressés en int8 APRÈS entraînement
# Aucune calibration nécessaire → très simple !
print("🔧 Application de la quantification dynamique...")
model_q1 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# ── Sauvegarder ──
os.makedirs("optimization/Q1_dynamic_quant", exist_ok=True)
SAVE = "optimization/Q1_dynamic_quant/model_q1.pt"
torch.save(model_q1.state_dict(), SAVE)

# ── Mesurer ──
t_train, t_test = creer_transforms()
train_loader, val_loader, test_loader, _, _, _ = creer_dataloaders(t_train, t_test)

taille_q1   = os.path.getsize(SAVE) / (1024*1024)
taille_base = os.path.getsize("baseline/model_baseline.pt") / (1024*1024)
_, acc, f1, _, _ = evaluer(model_q1, test_loader)
temps_ms, _ = mesurer_inference(model_q1, test_loader)

res = {
    "technique":    "Q1 - Quantification Dynamique",
    "accuracy":     round(acc, 4),
    "f1_score":     round(f1, 4),
    "taille_mo":    round(taille_q1, 2),
    "compression":  round(taille_base / taille_q1, 2),
    "inference_ms": temps_ms
}
with open("optimization/Q1_dynamic_quant/results_q1.json", "w") as f:
    json.dump(res, f, indent=2)

print(f"\n✅ [Q1] Résultats :")
print(f"   Accuracy    : {acc*100:.2f}%")
print(f"   Taille      : {taille_q1:.2f} Mo (baseline: {taille_base:.2f} Mo)")
print(f"   Compression : {taille_base/taille_q1:.2f}x")
print(f"   Inférence   : {temps_ms} ms/image")
