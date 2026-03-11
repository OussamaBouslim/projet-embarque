"""
=============================================================
 optimization/P3_magnitude/prune.py  — VERSION FINALE
 Elagage Magnitude iteratif — seulement couches pointwise
 (ignore les depthwise Conv2d fragiles de MobileNetV2)
=============================================================
"""
import os, sys, json, copy
import torch
import torch.nn.utils.prune as prune
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.train import construire_modele, evaluer, mesurer_inference
from dataset.preprocessing import creer_transforms, creer_dataloaders

DEVICE    = torch.device('cpu')
SEUIL_ACC = 0.90

print("📦 Chargement du modèle baseline...")
t_train, t_test = creer_transforms()
train_loader, val_loader, test_loader, _, _, _ = creer_dataloaders(t_train, t_test)

def appliquer_magnitude(model, taux):
    """Elague seulement les pointwise (1x1, groups=1)"""
    nb = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if module.groups == 1 and module.kernel_size == (1, 1):
                prune.l1_unstructured(module, name='weight', amount=taux)
                nb += 1
    return nb

def rendre_permanent(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try: prune.remove(module, 'weight')
            except ValueError: pass

print("✂️  Elagage magnitude iteratif (pointwise uniquement)...")
print(f"   Seuil minimum : {SEUIL_ACC*100:.0f}%\n")

taux_list      = [0.1, 0.2, 0.3, 0.4, 0.5]
meilleur_acc   = 0
meilleur_taux  = 0.1
meilleur_model = None
historique     = []

for taux in taux_list:
    model_test = construire_modele()
    model_test.load_state_dict(torch.load("baseline/model_baseline.pt", map_location=DEVICE))

    nb = appliquer_magnitude(model_test, taux)
    rendre_permanent(model_test)

    _, acc_val, _, _, _ = evaluer(model_test, val_loader)
    historique.append({"taux": taux, "accuracy_val": round(acc_val, 4), "couches": nb})
    statut = 'OK' if acc_val >= SEUIL_ACC else 'SOUS LE SEUIL'
    print(f"  Taux {int(taux*100):2d}% ({nb} couches) → Accuracy: {acc_val*100:.2f}%  [{statut}]")

    if acc_val >= SEUIL_ACC and acc_val >= meilleur_acc:
        meilleur_acc   = acc_val
        meilleur_taux  = taux
        meilleur_model = copy.deepcopy(model_test)
    elif acc_val < SEUIL_ACC:
        if meilleur_model is not None:
            print(f"  STOP → on garde taux {int(meilleur_taux*100)}%")
        break

if meilleur_model is None:
    print("  Fallback → taux 10%")
    meilleur_model = construire_modele()
    meilleur_model.load_state_dict(torch.load("baseline/model_baseline.pt", map_location=DEVICE))
    appliquer_magnitude(meilleur_model, 0.1)
    rendre_permanent(meilleur_model)
    meilleur_taux = 0.1

os.makedirs("optimization/P3_magnitude", exist_ok=True)
SAVE = "optimization/P3_magnitude/model_p3.pt"
torch.save(meilleur_model.state_dict(), SAVE)

taille_p3   = os.path.getsize(SAVE) / (1024*1024)
taille_base = os.path.getsize("baseline/model_baseline.pt") / (1024*1024)
_, acc, f1, _, _ = evaluer(meilleur_model, test_loader)
temps_ms, _ = mesurer_inference(meilleur_model, test_loader)

res = {
    "technique":     "P3 - Elagage Magnitude Iteratif (pointwise)",
    "note":          "Depthwise ignorees — pointwise uniquement",
    "meilleur_taux": meilleur_taux,
    "historique":    historique,
    "accuracy":      round(acc, 4),
    "f1_score":      round(f1, 4),
    "taille_mo":     round(taille_p3, 2),
    "compression":   round(taille_base / taille_p3, 2),
    "inference_ms":  temps_ms
}
with open("optimization/P3_magnitude/results_p3.json", "w") as f:
    json.dump(res, f, indent=2)

print(f"\n✅ [P3] Meilleur taux : {int(meilleur_taux*100)}%")
print(f"   Accuracy    : {acc*100:.2f}%")
print(f"   Taille      : {taille_p3:.2f} Mo")
print(f"   Compression : {taille_base/taille_p3:.2f}x")
print(f"   Inference   : {temps_ms} ms/image")
