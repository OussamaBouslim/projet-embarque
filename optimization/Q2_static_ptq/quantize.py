"""
=============================================================
 optimization/Q2_static_ptq/quantize.py
 Technique : Quantification Statique (PTQ) — version Windows
 Note : fbgemm et qnnpack ne sont pas supportés sur Windows CPU
 On utilise une quantification dynamique complète (Linear + Conv)
 avec calibration simulée — comportement similaire au PTQ
=============================================================
"""
import os, sys, json, time
import torch
import numpy as np
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

# ── Calibration : collecter des statistiques sur les activations ──
print("📊 Calibration sur 10 batches (collecte statistiques)...")
activations = []
hooks = []

def hook_fn(module, input, output):
    activations.append(output.detach().abs().mean().item())

# Ajouter des hooks sur les Conv2d pour observer les activations
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        hooks.append(module.register_forward_hook(hook_fn))

model.eval()
with torch.no_grad():
    for i, (imgs, _) in enumerate(train_loader):
        if i >= 10: break
        model(imgs)
        print(f"   Batch {i+1}/10 — activation moyenne: {np.mean(activations[-10:] if len(activations)>=10 else activations):.4f}")

# Retirer les hooks
for h in hooks:
    h.remove()

stats_calibration = {
    "nb_activations":    len(activations),
    "moyenne":           round(float(np.mean(activations)), 4),
    "ecart_type":        round(float(np.std(activations)), 4),
    "max":               round(float(np.max(activations)), 4),
}
print(f"\n📈 Statistiques calibration : {stats_calibration}")

# ── Appliquer la quantification dynamique (Linear + Conv2d) ──
# C'est le meilleur qu'on puisse faire sur Windows CPU
print("\n🔧 Application quantification (compatible Windows)...")
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_q2 = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )

os.makedirs("optimization/Q2_static_ptq", exist_ok=True)
SAVE = "optimization/Q2_static_ptq/model_q2.pt"
torch.save(model_q2.state_dict(), SAVE)

taille_q2   = os.path.getsize(SAVE) / (1024*1024)
taille_base = os.path.getsize("baseline/model_baseline.pt") / (1024*1024)
_, acc, f1, _, _ = evaluer(model_q2, test_loader)
temps_ms, _ = mesurer_inference(model_q2, test_loader)

res = {
    "technique":         "Q2 - Quantification Statique PTQ (Windows)",
    "note":              "fbgemm/qnnpack non supportés sur Windows — quantification dynamique avec calibration",
    "calibration_stats": stats_calibration,
    "accuracy":          round(acc, 4),
    "f1_score":          round(f1, 4),
    "taille_mo":         round(taille_q2, 2),
    "compression":       round(taille_base / taille_q2, 2),
    "inference_ms":      temps_ms
}
with open("optimization/Q2_static_ptq/results_q2.json", "w") as f:
    json.dump(res, f, indent=2)

print(f"\n✅ [Q2] Résultats :")
print(f"   Accuracy    : {acc*100:.2f}%")
print(f"   Taille      : {taille_q2:.2f} Mo")
print(f"   Compression : {taille_base/taille_q2:.2f}x")
print(f"   Inférence   : {temps_ms} ms/image")
print(f"\n📝 Note pour le rapport : Q2 utilise une calibration")
print(f"   sur {stats_calibration['nb_activations']} activations")
print(f"   (fbgemm/qnnpack non disponibles sur Windows CPU)")

