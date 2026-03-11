"""
=============================================================
 optimization/P1_unstructured/prune.py  — VERSION CORRIGEE
 Taux adaptatif : cherche le meilleur taux (10%→20%→30%)
 en repartant du modele original à chaque fois
=============================================================
"""
import os, sys, json, copy
import torch
import torch.nn.utils.prune as prune
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.train import construire_modele, evaluer, mesurer_inference
from dataset.preprocessing import creer_transforms, creer_dataloaders

DEVICE    = torch.device('cpu')
SEUIL_ACC = 0.90   # 90% minimum

print("📦 Chargement du modèle baseline...")
t_train, t_test = creer_transforms()
train_loader, val_loader, test_loader, _, _, _ = creer_dataloaders(t_train, t_test)

print("✂️  Elagage non structure — recherche du meilleur taux...")
taux_list      = [0.1, 0.2, 0.3]
meilleur_acc   = 0
meilleur_taux  = 0.1
meilleur_model = None
historique     = []

for taux in taux_list:
    # Repartir du modele original à chaque taux
    model_test = construire_modele()
    model_test.load_state_dict(torch.load("baseline/model_baseline.pt", map_location=DEVICE))

    for name, module in model_test.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=taux)

    for name, module in model_test.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try: prune.remove(module, 'weight')
            except ValueError: pass

    _, acc_val, _, _, _ = evaluer(model_test, val_loader)
    historique.append({"taux": taux, "accuracy_val": round(acc_val, 4)})
    statut = 'OK' if acc_val >= SEUIL_ACC else 'SOUS LE SEUIL'
    print(f"  Taux {int(taux*100):2d}% → Accuracy val: {acc_val*100:.2f}%  [{statut}]")

    if acc_val >= SEUIL_ACC and acc_val >= meilleur_acc:
        meilleur_acc   = acc_val
        meilleur_taux  = taux
        meilleur_model = copy.deepcopy(model_test)
    elif acc_val < SEUIL_ACC:
        if meilleur_model is not None:
            print(f"  STOP → on garde taux {int(meilleur_taux*100)}%")
        break

if meilleur_model is None:
    print("  Aucun taux OK → utilisation de 10%")
    meilleur_model = construire_modele()
    meilleur_model.load_state_dict(torch.load("baseline/model_baseline.pt", map_location=DEVICE))
    for name, module in meilleur_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.1)
    for name, module in meilleur_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try: prune.remove(module, 'weight')
            except ValueError: pass
    meilleur_taux = 0.1

os.makedirs("optimization/P1_unstructured", exist_ok=True)
SAVE = "optimization/P1_unstructured/model_p1.pt"
torch.save(meilleur_model.state_dict(), SAVE)

taille_p1   = os.path.getsize(SAVE) / (1024*1024)
taille_base = os.path.getsize("baseline/model_baseline.pt") / (1024*1024)
_, acc, f1, _, _ = evaluer(meilleur_model, test_loader)
temps_ms, _ = mesurer_inference(meilleur_model, test_loader)

res = {
    "technique":     "P1 - Elagage Non Structure",
    "meilleur_taux": meilleur_taux,
    "historique":    historique,
    "accuracy":      round(acc, 4),
    "f1_score":      round(f1, 4),
    "taille_mo":     round(taille_p1, 2),
    "compression":   round(taille_base / taille_p1, 2),
    "inference_ms":  temps_ms
}
with open("optimization/P1_unstructured/results_p1.json", "w") as f:
    json.dump(res, f, indent=2)

print(f"\n✅ [P1] Meilleur taux : {int(meilleur_taux*100)}%")
print(f"   Accuracy    : {acc*100:.2f}%")
print(f"   Taille      : {taille_p1:.2f} Mo")
print(f"   Compression : {taille_base/taille_p1:.2f}x")
print(f"   Inference   : {temps_ms} ms/image")
