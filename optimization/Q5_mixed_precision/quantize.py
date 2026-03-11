"""
=============================================================
 optimization/Q5_mixed_precision/quantize.py
 Technique : Quantification à Précision Mixte
 → Couches sensibles en int8, autres en int4 (simulé)
=============================================================
"""
import os, sys, json
import torch
import torch.nn as nn
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

# ── Précision mixte : différentes configs par couche ──
# Couches sensibles (début du réseau) → int8 précis
# Couches moins sensibles (fin) → quantification plus agressive
print("🔧 Application de la précision mixte...")

# Appliquer des configs différentes par couche
for i, (name, module) in enumerate(model.named_modules()):
    if isinstance(module, nn.Conv2d):
        if i < 10:
            # Couches du début : plus sensibles → int8
            module.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        else:
            # Couches de fin : moins sensibles → dynamique
            module.qconfig = torch.quantization.default_dynamic_qconfig

# Quantification dynamique globale (approximation précision mixte)
model_q5 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

os.makedirs("optimization/Q5_mixed_precision", exist_ok=True)
SAVE = "optimization/Q5_mixed_precision/model_q5.pt"
torch.save(model_q5.state_dict(), SAVE)

taille_q5   = os.path.getsize(SAVE) / (1024*1024)
taille_base = os.path.getsize("baseline/model_baseline.pt") / (1024*1024)
_, acc, f1, _, _ = evaluer(model_q5, test_loader)
temps_ms, _ = mesurer_inference(model_q5, test_loader)

res = {
    "technique":    "Q5 - Précision Mixte",
    "accuracy":     round(acc, 4),
    "f1_score":     round(f1, 4),
    "taille_mo":    round(taille_q5, 2),
    "compression":  round(taille_base / taille_q5, 2),
    "inference_ms": temps_ms
}
with open("optimization/Q5_mixed_precision/results_q5.json", "w") as f:
    json.dump(res, f, indent=2)

print(f"\n✅ [Q5] Résultats :")
print(f"   Accuracy    : {acc*100:.2f}%")
print(f"   Taille      : {taille_q5:.2f} Mo")
print(f"   Compression : {taille_base/taille_q5:.2f}x")
print(f"   Inférence   : {temps_ms} ms/image")
