"""
=============================================================
 optimization/Q2_static_ptq/quantize.py
 Technique : Quantification Statique (PTQ)
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

# ── Quantification statique : nécessite une calibration ──
print("🔧 Préparation de la quantification statique...")
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibration : passer ~300 images pour apprendre les plages de valeurs
print("📊 Calibration sur 10 batches...")
model.eval()
with torch.no_grad():
    for i, (imgs, _) in enumerate(train_loader):
        if i >= 10: break
        model(imgs)

# Convertir en modèle quantifié final
torch.quantization.convert(model, inplace=True)

os.makedirs("optimization/Q2_static_ptq", exist_ok=True)
SAVE = "optimization/Q2_static_ptq/model_q2.pt"
torch.save(model.state_dict(), SAVE)

taille_q2   = os.path.getsize(SAVE) / (1024*1024)
taille_base = os.path.getsize("baseline/model_baseline.pt") / (1024*1024)
_, acc, f1, _, _ = evaluer(model, test_loader)
temps_ms, _ = mesurer_inference(model, test_loader)

res = {
    "technique":    "Q2 - Quantification Statique PTQ",
    "accuracy":     round(acc, 4),
    "f1_score":     round(f1, 4),
    "taille_mo":    round(taille_q2, 2),
    "compression":  round(taille_base / taille_q2, 2),
    "inference_ms": temps_ms
}
with open("optimization/Q2_static_ptq/results_q2.json", "w") as f:
    json.dump(res, f, indent=2)

print(f"\n✅ [Q2] Résultats :")
print(f"   Accuracy    : {acc*100:.2f}%")
print(f"   Taille      : {taille_q2:.2f} Mo")
print(f"   Compression : {taille_base/taille_q2:.2f}x")
print(f"   Inférence   : {temps_ms} ms/image")
