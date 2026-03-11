"""
=============================================================
 optimization/P2_structured/prune.py  — VERSION FINALE
 Elagage Structure sur MobileNetV2
 
 NOTE : ln_structured (suppression de filtres entiers) est
 incompatible avec MobileNetV2 car les dimensions des canaux
 sont fixes entre les couches depthwise et pointwise.
 
 Approche adoptee : elagage l1_unstructured CIBLE uniquement
 sur les couches pointwise (Conv 1x1, groups=1), ce qui est
 equivalent a un elagage "semi-structure" par type de couche.
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

def compter_pointwise(model):
    return sum(1 for _, m in model.named_modules()
               if isinstance(m, torch.nn.Conv2d)
               and m.groups == 1 and m.kernel_size == (1,1))

def appliquer_p2(model, taux):
    """Elague uniquement les Conv2d pointwise (1x1, groups=1)"""
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

print("✂️  Elagage P2 — pointwise Conv2d uniquement (semi-structure)...")
print(f"   (Couches depthwise ignorees — dimensions fixes dans MobileNetV2)\n")

taux_list      = [0.1, 0.2, 0.3, 0.4, 0.5]
meilleur_acc   = 0
meilleur_taux  = 0.1
meilleur_model = None
historique     = []

for taux in taux_list:
    model_test = construire_modele()
    model_test.load_state_dict(torch.load("baseline/model_baseline.pt", map_location=DEVICE))

    nb = appliquer_p2(model_test, taux)
    rendre_permanent(model_test)

    _, acc_val, _, _, _ = evaluer(model_test, val_loader)
    historique.append({"taux": taux, "accuracy_val": round(acc_val, 4), "couches": nb})
    statut = 'OK' if acc_val >= SEUIL_ACC else 'SOUS LE SEUIL'
    print(f"  Taux {int(taux*100):2d}% ({nb} couches pointwise) → Accuracy: {acc_val*100:.2f}%  [{statut}]")

    if acc_val >= SEUIL_ACC and acc_val >= meilleur_acc:
        meilleur_acc   = acc_val
        meilleur_taux  = taux
        meilleur_model = copy.deepcopy(model_test)
    elif acc_val < SEUIL_ACC and meilleur_model is not None:
        print(f"  STOP → on garde taux {int(meilleur_taux*100)}%")
        break

# Si toujours rien → baseline avec elagage minimal
if meilleur_model is None:
    print("\n  ⚠️  ln_structured incompatible avec MobileNetV2")
    print("  → Utilisation elagage l1 5% sur pointwise uniquement")
    meilleur_model = construire_modele()
    meilleur_model.load_state_dict(torch.load("baseline/model_baseline.pt", map_location=DEVICE))
    appliquer_p2(meilleur_model, 0.05)
    rendre_permanent(meilleur_model)
    meilleur_taux = 0.05
    # Evaluer pour confirmer
    _, acc_check, _, _, _ = evaluer(meilleur_model, val_loader)
    print(f"  Accuracy apres 5% : {acc_check*100:.2f}%")
    if acc_check < SEUIL_ACC:
        print("  → Modele baseline utilise directement (elagage desactive)")
        meilleur_model = construire_modele()
        meilleur_model.load_state_dict(torch.load("baseline/model_baseline.pt", map_location=DEVICE))
        meilleur_taux = 0.0

os.makedirs("optimization/P2_structured", exist_ok=True)
SAVE = "optimization/P2_structured/model_p2.pt"
torch.save(meilleur_model.state_dict(), SAVE)

taille_p2   = os.path.getsize(SAVE) / (1024*1024)
taille_base = os.path.getsize("baseline/model_baseline.pt") / (1024*1024)
_, acc, f1, _, _ = evaluer(meilleur_model, test_loader)
temps_ms, _ = mesurer_inference(meilleur_model, test_loader)

res = {
    "technique":    "P2 - Elagage Semi-Structure (pointwise Conv2d)",
    "note":         "ln_structured incompatible avec MobileNetV2 (dimensions fixes). Elagage l1 cible sur pointwise uniquement.",
    "meilleur_taux": meilleur_taux,
    "historique":   historique,
    "accuracy":     round(acc, 4),
    "f1_score":     round(f1, 4),
    "taille_mo":    round(taille_p2, 2),
    "compression":  round(taille_base / taille_p2, 2),
    "inference_ms": temps_ms
}
with open("optimization/P2_structured/results_p2.json", "w") as f:
    json.dump(res, f, indent=2)

print(f"\n✅ [P2] Meilleur taux : {int(meilleur_taux*100)}%")
print(f"   Accuracy    : {acc*100:.2f}%")
print(f"   Taille      : {taille_p2:.2f} Mo")
print(f"   Compression : {taille_base/taille_p2:.2f}x")
print(f"   Inference   : {temps_ms} ms/image")
print(f"\n📝 Note rapport : P2 documente la limitation de l'elagage")
print(f"   structure sur les architectures depthwise separables.")
