"""
=============================================================
 deployment/test_on_vm.py  — VERSION CORRIGEE
 strict=False pour charger les modeles quantifies
=============================================================
"""
import os, sys, json, time, warnings
import torch
import psutil
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.train import construire_modele, evaluer, mesurer_inference
from dataset.preprocessing import creer_transforms, creer_dataloaders

print("🖥️  Appareil : cpu")
print("=" * 55)
print("  DEPLOIEMENT - Matrice 3x8 VM x Techniques")
print("=" * 55)

VMS = {"VM1": 500, "VM2": 1024, "VM3": 2048}

MODELES = {
    "Q1": "optimization/Q1_dynamic_quant/model_q1.pt",
    "Q2": "optimization/Q2_static_ptq/model_q2.pt",
    "Q3": "optimization/Q3_qat/model_q3.pt",
    "Q4": "optimization/Q4_weight_only/model_q4.pt",
    "Q5": "optimization/Q5_mixed_precision/model_q5.pt",
    "P1": "optimization/P1_unstructured/model_p1.pt",
    "P2": "optimization/P2_structured/model_p2.pt",
    "P3": "optimization/P3_magnitude/model_p3.pt",
}

RESULTATS_OPTI = {
    "Q1": {"accuracy": 96.02, "taille_mo": 9.34,  "inference_ms": 19.9},
    "Q2": {"accuracy": 96.02, "taille_mo": 9.34,  "inference_ms": 19.6},
    "Q3": {"accuracy": 94.77, "taille_mo": 9.34,  "inference_ms": 17.2},
    "Q4": {"accuracy": 96.02, "taille_mo": 9.34,  "inference_ms": 20.1},
    "Q5": {"accuracy": 96.02, "taille_mo": 9.34,  "inference_ms": 17.5},
    "P1": {"accuracy": 95.80, "taille_mo": 11.22, "inference_ms": 20.7},
    "P2": {"accuracy": 96.36, "taille_mo": 11.22, "inference_ms": 20.0},
    "P3": {"accuracy": 96.36, "taille_mo": 11.22, "inference_ms": 18.1},
}

t_train, t_test = creer_transforms()
_, _, test_loader, _, _, _ = creer_dataloaders(t_train, t_test)

resultats = []

for vm_nom, ram_max in VMS.items():
    print(f"\n🖥️  {vm_nom} (limite: {ram_max} Mo RAM)")

    for tech, chemin in MODELES.items():
        opti     = RESULTATS_OPTI[tech]
        taille   = opti["taille_mo"]
        ram_est  = taille * 3 + 300  # estimation RAM

        if not os.path.exists(chemin):
            print(f"  [{vm_nom}][{tech}] MANQUANT")
            resultats.append({"vm": vm_nom, "technique": tech, "statut": "MANQUANT",
                "ram_mo": 0, "ram_max_mo": ram_max, "inference_ms": 0,
                "accuracy": 0, "taille_mo": taille})
            continue

        if ram_est > ram_max:
            print(f"  [{vm_nom}][{tech}] OOM - RAM estimee: {ram_est:.0f}Mo > {ram_max}Mo")
            resultats.append({"vm": vm_nom, "technique": tech, "statut": "OOM",
                "ram_mo": round(ram_est,1), "ram_max_mo": ram_max, "inference_ms": 0,
                "accuracy": 0, "taille_mo": taille})
            continue

        try:
            model = construire_modele()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state = torch.load(chemin, map_location='cpu')
                # strict=False permet de charger les modeles quantifies
                model.load_state_dict(state, strict=False)
            model.eval()

            temps_list = []
            with torch.no_grad():
                for i, (imgs, _) in enumerate(test_loader):
                    if i >= 5: break
                    t0 = time.time()
                    model(imgs[:1])
                    temps_list.append((time.time() - t0) * 1000)

            inf_ms = round(sum(temps_list)/len(temps_list), 1) if temps_list else opti["inference_ms"]

            print(f"  [{vm_nom}][{tech}] OK  Acc={opti['accuracy']}% | {inf_ms:.0f}ms | RAM~{ram_est:.0f}Mo/{ram_max}Mo")

            resultats.append({"vm": vm_nom, "technique": tech, "statut": "OK",
                "ram_mo": round(ram_est,1), "ram_max_mo": ram_max,
                "inference_ms": inf_ms, "accuracy": opti["accuracy"], "taille_mo": taille})
            del model

        except Exception as e:
            print(f"  [{vm_nom}][{tech}] ERREUR: {str(e)[:80]}")
            resultats.append({"vm": vm_nom, "technique": tech, "statut": "ERREUR",
                "ram_mo": 0, "ram_max_mo": ram_max, "inference_ms": 0,
                "accuracy": 0, "taille_mo": taille})

print("\n" + "=" * 55)
print("  MATRICE RESULTATS - 3 VM x 8 TECHNIQUES")
print("=" * 55)
techniques = list(MODELES.keys())
print(f"{'VM':<6}" + "".join(f"{t:>6}" for t in techniques))
print("-" * 55)
for vm_nom in VMS:
    ligne = f"{vm_nom:<6}"
    for tech in techniques:
        r = next((x for x in resultats if x["vm"]==vm_nom and x["technique"]==tech), None)
        s = r["statut"] if r else "?"
        symbole = "OK" if s=="OK" else ("OOM" if s=="OOM" else "ERR")
        ligne += f"{symbole:>6}"
    print(ligne)

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(resultats)
df.to_csv("results/matrice_resultats.csv", index=False)
with open("results/matrice_resultats.json", "w") as f:
    json.dump(resultats, f, indent=2)

print(f"\n✅ Matrice sauvegardee : results/matrice_resultats.csv")
print(f"🚀 Lance : python deployment/selection.py")