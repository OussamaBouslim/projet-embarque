"""
=============================================================
 optimization/P1_unstructured/prune.py
 Technique : Élagage Non Structuré (30%)
=============================================================
"""
import os, sys, json
import torch
import torch.nn.utils.prune as prune
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.train import construire_modele, evaluer, mesurer_inference
from dataset.preprocessing import creer_transforms, creer_dataloaders

DEVICE = torch.device('cpu')
TAUX   = 0.3   # 30% des poids → zéro

print("📦 Chargement du modèle baseline...")
model = construire_modele()
model.load_state_dict(torch.load("baseline/model_baseline.pt", map_location=DEVICE))

t_train, t_test = creer_transforms()
train_loader, val_loader, test_loader, _, _, _ = creer_dataloaders(t_train, t_test)

# ── Élagage non structuré : poids individuels mis à zéro ──
print(f"✂️  Élagage non structuré ({int(TAUX*100)}%)...")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=TAUX)

# Rendre l'élagage permanent
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')

os.makedirs("optimization/P1_unstructured", exist_ok=True)
SAVE = "optimization/P1_unstructured/model_p1.pt"
torch.save(model.state_dict(), SAVE)

taille_p1   = os.path.getsize(SAVE) / (1024*1024)
taille_base = os.path.getsize("baseline/model_baseline.pt") / (1024*1024)
_, acc, f1, _, _ = evaluer(model, test_loader)
temps_ms, _ = mesurer_inference(model, test_loader)

res = {
    "technique":    "P1 - Élagage Non Structuré",
    "taux_elagage": TAUX,
    "accuracy":     round(acc, 4),
    "f1_score":     round(f1, 4),
    "taille_mo":    round(taille_p1, 2),
    "compression":  round(taille_base / taille_p1, 2),
    "inference_ms": temps_ms
}
with open("optimization/P1_unstructured/results_p1.json", "w") as f:
    json.dump(res, f, indent=2)

print(f"\n✅ [P1] Résultats :")
print(f"   Accuracy    : {acc*100:.2f}%")
print(f"   Taille      : {taille_p1:.2f} Mo")
print(f"   Compression : {taille_base/taille_p1:.2f}x")
print(f"   Inférence   : {temps_ms} ms/image")
