"""
=============================================================
 optimization/Q4_weight_only/quantize.py
 Technique : Quantification Weight-Only (int8)
=============================================================
"""
import os, sys, json
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.train import construire_modele, evaluer, mesurer_inference
from dataset.preprocessing import creer_transforms, creer_dataloaders

DEVICE = torch.device('cpu')

print("📦 Chargement du modèle baseline...")
model = construire_modele()
model.load_state_dict(torch.load("baseline/model_baseline.pt", map_location=DEVICE))
model.eval()

t_train, t_test = creer_transforms()
train_loader, val_loader, test_loader, _, _, _ = creer_dataloaders(t_train, t_test)

# ── Weight-Only : quantifier SEULEMENT les poids (pas les activations) ──
# Les activations restent en float32
print("🔧 Application Weight-Only (poids en int8)...")
model_q4 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},   # Seulement les couches Linear
    dtype=torch.qint8
)

os.makedirs("optimization/Q4_weight_only", exist_ok=True)
SAVE = "optimization/Q4_weight_only/model_q4.pt"
torch.save(model_q4.state_dict(), SAVE)

taille_q4   = os.path.getsize(SAVE) / (1024*1024)
taille_base = os.path.getsize("baseline/model_baseline.pt") / (1024*1024)
_, acc, f1, _, _ = evaluer(model_q4, test_loader)
temps_ms, _ = mesurer_inference(model_q4, test_loader)

res = {
    "technique":    "Q4 - Weight-Only",
    "accuracy":     round(acc, 4),
    "f1_score":     round(f1, 4),
    "taille_mo":    round(taille_q4, 2),
    "compression":  round(taille_base / taille_q4, 2),
    "inference_ms": temps_ms
}
with open("optimization/Q4_weight_only/results_q4.json", "w") as f:
    json.dump(res, f, indent=2)

print(f"\n✅ [Q4] Résultats :")
print(f"   Accuracy    : {acc*100:.2f}%")
print(f"   Taille      : {taille_q4:.2f} Mo")
print(f"   Compression : {taille_base/taille_q4:.2f}x")
print(f"   Inférence   : {temps_ms} ms/image")
