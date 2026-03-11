"""
=============================================================
 optimization/P2_structured/prune.py
 Technique : Élagage Structuré (filtres entiers supprimés)
=============================================================
"""
import os, sys, json
import torch
import torch.nn.utils.prune as prune
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.train import construire_modele, evaluer, mesurer_inference
from dataset.preprocessing import creer_transforms, creer_dataloaders

DEVICE = torch.device('cpu')
TAUX   = 0.3

print("📦 Chargement du modèle baseline...")
model = construire_modele()
model.load_state_dict(torch.load("baseline/model_baseline.pt", map_location=DEVICE))

t_train, t_test = creer_transforms()
train_loader, val_loader, test_loader, _, _, _ = creer_dataloaders(t_train, t_test)

# ── Élagage structuré : supprimer des FILTRES entiers ──
# → Le modèle devient vraiment plus petit et plus rapide
print(f"✂️  Élagage structuré ({int(TAUX*100)}% des filtres)...")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) and module.out_channels > 8:
        prune.ln_structured(
            module,
            name='weight',
            amount=TAUX,
            n=2,    # Norme L2
            dim=0   # dim=0 = filtres de sortie
        )

# Rendre permanent
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) and module.out_channels > 8:
        try:
            prune.remove(module, 'weight')
        except ValueError:
            pass

os.makedirs("optimization/P2_structured", exist_ok=True)
SAVE = "optimization/P2_structured/model_p2.pt"
torch.save(model.state_dict(), SAVE)

taille_p2   = os.path.getsize(SAVE) / (1024*1024)
taille_base = os.path.getsize("baseline/model_baseline.pt") / (1024*1024)
_, acc, f1, _, _ = evaluer(model, test_loader)
temps_ms, _ = mesurer_inference(model, test_loader)

res = {
    "technique":    "P2 - Élagage Structuré",
    "taux_elagage": TAUX,
    "accuracy":     round(acc, 4),
    "f1_score":     round(f1, 4),
    "taille_mo":    round(taille_p2, 2),
    "compression":  round(taille_base / taille_p2, 2),
    "inference_ms": temps_ms
}
with open("optimization/P2_structured/results_p2.json", "w") as f:
    json.dump(res, f, indent=2)

print(f"\n✅ [P2] Résultats :")
print(f"   Accuracy    : {acc*100:.2f}%")
print(f"   Taille      : {taille_p2:.2f} Mo")
print(f"   Compression : {taille_base/taille_p2:.2f}x")
print(f"   Inférence   : {temps_ms} ms/image")
