"""
=============================================================
 optimization/P3_magnitude/prune.py
 Technique : Élagage par Magnitude (itératif)
=============================================================
"""
import os, sys, json
import torch
import torch.nn.utils.prune as prune
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.train import construire_modele, evaluer, mesurer_inference
from dataset.preprocessing import creer_transforms, creer_dataloaders

DEVICE = torch.device('cpu')

print("📦 Chargement du modèle baseline...")
model = construire_modele()
model.load_state_dict(torch.load("baseline/model_baseline.pt", map_location=DEVICE))

t_train, t_test = creer_transforms()
train_loader, val_loader, test_loader, _, _, _ = creer_dataloaders(t_train, t_test)

# ── Élagage par magnitude : itératif 10% → 20% → 30% ──
# À chaque étape, on supprime les poids les plus faibles
print("✂️  Élagage par magnitude (itératif)...")
taux_list = [0.1, 0.2, 0.3]

for taux in taux_list:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=taux)
    _, acc_val, _, _, _ = evaluer(model, val_loader)
    print(f"  Après {int(taux*100)}% → Accuracy val: {acc_val*100:.2f}%")

# Rendre permanent
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        try:
            prune.remove(module, 'weight')
        except ValueError:
            pass

os.makedirs("optimization/P3_magnitude", exist_ok=True)
SAVE = "optimization/P3_magnitude/model_p3.pt"
torch.save(model.state_dict(), SAVE)

taille_p3   = os.path.getsize(SAVE) / (1024*1024)
taille_base = os.path.getsize("baseline/model_baseline.pt") / (1024*1024)
_, acc, f1, _, _ = evaluer(model, test_loader)
temps_ms, _ = mesurer_inference(model, test_loader)

res = {
    "technique":    "P3 - Élagage Magnitude",
    "taux_final":   0.3,
    "accuracy":     round(acc, 4),
    "f1_score":     round(f1, 4),
    "taille_mo":    round(taille_p3, 2),
    "compression":  round(taille_base / taille_p3, 2),
    "inference_ms": temps_ms
}
with open("optimization/P3_magnitude/results_p3.json", "w") as f:
    json.dump(res, f, indent=2)

print(f"\n✅ [P3] Résultats :")
print(f"   Accuracy    : {acc*100:.2f}%")
print(f"   Taille      : {taille_p3:.2f} Mo")
print(f"   Compression : {taille_base/taille_p3:.2f}x")
print(f"   Inférence   : {temps_ms} ms/image")
