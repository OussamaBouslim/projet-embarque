"""
=============================================================
 deployment/test_on_vm.py
 Teste chaque modèle optimisé sur chaque VM Docker
 et génère la matrice de résultats 3x8
=============================================================
"""
import os, sys, json, time
import numpy as np
import torch
import psutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.train import construire_modele, evaluer
from dataset.preprocessing import creer_transforms, creer_dataloaders

DEVICE = torch.device('cpu')

# ── Liste des techniques et leurs modèles ──
TECHNIQUES = {
    "Q1": "optimization/Q1_dynamic_quant/model_q1.pt",
    "Q2": "optimization/Q2_static_ptq/model_q2.pt",
    "Q3": "optimization/Q3_qat/model_q3.pt",
    "Q4": "optimization/Q4_weight_only/model_q4.pt",
    "Q5": "optimization/Q5_mixed_precision/model_q5.pt",
    "P1": "optimization/P1_unstructured/model_p1.pt",
    "P2": "optimization/P2_structured/model_p2.pt",
    "P3": "optimization/P3_magnitude/model_p3.pt",
}

# ── Limites RAM des VM ──
VM_LIMITES = {
    "VM1": 500,   # Mo
    "VM2": 1024,  # Mo
    "VM3": 2048,  # Mo
}

def tester_modele(technique_id, chemin_modele, test_loader, vm_id):
    """Teste un modèle sur une VM simulée."""

    # Vérifier si le fichier existe
    if not os.path.exists(chemin_modele):
        return {"vm": vm_id, "technique": technique_id, "status": "MANQUANT"}

    # Vérifier si le modèle rentre en RAM
    taille_mo = os.path.getsize(chemin_modele) / (1024*1024)
    ram_limite = VM_LIMITES[vm_id]
    if taille_mo * 3 > ram_limite:   # facteur 3 = overhead runtime
        print(f"  [{vm_id}][{technique_id}] ❌ OOM ({taille_mo:.1f}Mo > {ram_limite//3}Mo dispo)")
        return {"vm": vm_id, "technique": technique_id, "status": "OOM",
                "taille_mo": round(taille_mo, 2)}

    # Charger et tester
    try:
        model = construire_modele()
        model.load_state_dict(torch.load(chemin_modele, map_location=DEVICE))
        model.eval()

        # 10 inférences
        temps_list = []
        with torch.no_grad():
            for i, (imgs, _) in enumerate(test_loader):
                if i >= 10: break
                t = time.perf_counter()
                model(imgs)
                temps_list.append((time.perf_counter()-t)*1000)

        _, acc, f1, _, _ = evaluer(model, test_loader)
        ram_utilisee = psutil.Process().memory_info().rss / (1024**2)
        cpu_pct      = psutil.cpu_percent(interval=0.5)

        res = {
            "vm":         vm_id,
            "technique":  technique_id,
            "status":     "OK",
            "accuracy":   round(acc, 4),
            "f1_score":   round(f1, 4),
            "taille_mo":  round(taille_mo, 2),
            "temps_ms":   round(np.mean(temps_list), 2),
            "temps_std":  round(np.std(temps_list), 2),
            "ram_mo":     round(ram_utilisee, 1),
            "cpu_pct":    round(cpu_pct, 1),
        }
        print(f"  [{vm_id}][{technique_id}] ✅ Acc={acc*100:.1f}% | {np.mean(temps_list):.0f}ms | RAM={ram_utilisee:.0f}Mo")
        return res

    except Exception as e:
        print(f"  [{vm_id}][{technique_id}] ❌ Erreur: {e}")
        return {"vm": vm_id, "technique": technique_id, "status": f"ERREUR: {e}"}


if __name__ == "__main__":
    print("="*55)
    print("  DÉPLOIEMENT — Matrice 3×8 VM × Techniques")
    print("="*55)

    t_train, t_test = creer_transforms()
    _, _, test_loader, _, _, _ = creer_dataloaders(t_train, t_test)

    tous_resultats = []
    for vm_id in ["VM1", "VM2", "VM3"]:
        print(f"\n🖥️  {vm_id} (limite: {VM_LIMITES[vm_id]} Mo RAM)")
        for tech_id, chemin in TECHNIQUES.items():
            res = tester_modele(tech_id, chemin, test_loader, vm_id)
            tous_resultats.append(res)

    # Sauvegarder en CSV
    os.makedirs("results", exist_ok=True)
    import csv
    champs = ["vm","technique","status","accuracy","f1_score",
              "taille_mo","temps_ms","temps_std","ram_mo","cpu_pct"]
    with open("results/matrice_resultats.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=champs)
        writer.writeheader()
        for r in tous_resultats:
            row = {c: r.get(c, "") for c in champs}
            writer.writerow(row)

    print(f"\n✅ Matrice sauvegardée : results/matrice_resultats.csv")
    print("🚀 Lance : python deployment/selection.py")
