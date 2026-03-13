"""
=============================================================
 collective/orchestrateur.py  — VERSION CORRIGEE
 Intelligence collective : vote pondéré des 3 VM
=============================================================
"""
import os, sys, json, time, warnings
import torch
import torch.nn.functional as F
import psutil
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.train import construire_modele, evaluer
from dataset.preprocessing import creer_transforms, creer_dataloaders

DEVICE  = torch.device('cpu')
CLASSES = ['NORMAL', 'PNEUMONIA']

# ── Précisions réelles obtenues (résultats optimisation) ──
PRECISIONS_VM = {
    'VM1': 0.9602,   # Q2 → 96.02%
    'VM2': 0.9602,   # Q4 → 96.02%
    'VM3': 0.9602,   # Q5 → 96.02%
}

# ── Meilleurs modèles sélectionnés par selection.py ──
MODELES_VM = {
    'VM1': 'optimization/Q2_static_ptq/model_q2.pt',    # Score 0.957
    'VM2': 'optimization/Q4_weight_only/model_q4.pt',   # Score 0.799
    'VM3': 'optimization/Q5_mixed_precision/model_q5.pt', # Score 0.872
}


def charger_modeles():
    """Charge les 3 modèles en mémoire avec strict=False."""
    modeles = {}
    for vm_id, chemin in MODELES_VM.items():
        if os.path.exists(chemin):
            m = construire_modele()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state = torch.load(chemin, map_location=DEVICE)
                # strict=False nécessaire pour les modèles quantifiés (Q1-Q5)
                m.load_state_dict(state, strict=False)
            m.eval()
            modeles[vm_id] = m
            print(f"  ✅ {vm_id} chargé : {chemin}")
        else:
            print(f"  ⚠️  {vm_id} : modèle introuvable ({chemin})")
    return modeles


def inferer(model, image):
    """Retourne la prédiction et la confiance."""
    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        probas = F.softmax(logits, dim=1)
        conf, pred = probas.max(dim=1)
    return pred.item(), conf.item(), probas.squeeze().tolist()


def vote_pondere(resultats_vms):
    """
    Combine les votes : poids = confiance × précision_historique
    """
    scores = defaultdict(float)
    for r in resultats_vms:
        poids = r['confiance'] * PRECISIONS_VM[r['vm']]
        scores[r['pred']] += poids
    total   = sum(scores.values())
    gagnant = max(scores, key=scores.get)
    conf_collective = scores[gagnant] / total if total > 0 else 0
    return gagnant, conf_collective


def orchestrer(image, modeles):
    """Pipeline complet : inférence → vote → validation → résultat."""
    resultats = []
    for vm_id, model in modeles.items():
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent
        # Seuils élevés pour simulation locale (pas de vraie VM isolée)
        if cpu > 99 or ram > 99:
            print(f"  ⚠️  {vm_id} surchargée (CPU={cpu}% RAM={ram}%) → ignorée")
            continue
        print(f"  [{vm_id}] CPU={cpu:.0f}% RAM={ram:.0f}%")

        pred, conf, probas = inferer(model, image)
        resultats.append({
            'vm': vm_id, 'pred': pred,
            'confiance': conf, 'probas': probas
        })
        print(f"  [{vm_id}] → {CLASSES[pred]} ({conf:.0%})")

    if not resultats:
        return None, 0, []

    diagnostic, conf_collective = vote_pondere(resultats)

    if conf_collective < 0.70:
        print(f"  ⚠️  Confiance faible ({conf_collective:.0%}) → Validation renforcée !")

    predictions = [r['pred'] for r in resultats]
    consensus   = all(p == predictions[0] for p in predictions)

    print(f"\n  🏥 Diagnostic final : {CLASSES[diagnostic]}")
    print(f"  📊 Confiance collective : {conf_collective:.0%}")
    print(f"  🤝 Consensus : {'✅ OUI' if consensus else '❌ NON'}")
    return diagnostic, conf_collective, resultats


if __name__ == "__main__":
    print("🖥️  Appareil : cpu")
    print("=" * 55)
    print("  INTELLIGENCE COLLECTIVE — Test sur 10 exemples")
    print("=" * 55)

    modeles = charger_modeles()

    if not modeles:
        print("\n❌ Aucun modèle chargé ! Vérifie les chemins.")
        exit()

    t_train, t_test = creer_transforms()
    _, _, test_loader, _, _, test_ds = creer_dataloaders(t_train, t_test)

    correct_collectif  = 0
    correct_individuel = {'VM1': 0, 'VM2': 0, 'VM3': 0}
    nb_consensus = 0
    nb_test      = 10

    resultats_tous = []
    for i, (imgs, labels) in enumerate(test_loader):
        if i >= nb_test: break
        img, label = imgs[0], labels[0].item()

        print(f"\n{'─'*50}")
        print(f"  Patient {i+1} | Vérité : {CLASSES[label]}")
        print(f"{'─'*50}")

        diagnostic, conf, votes = orchestrer(img, modeles)

        if diagnostic is not None:
            if diagnostic == label:
                correct_collectif += 1
            predictions = [r['pred'] for r in votes]
            if all(p == predictions[0] for p in predictions):
                nb_consensus += 1
            for r in votes:
                if r['pred'] == label:
                    correct_individuel[r['vm']] += 1

        resultats_tous.append({
            'patient': i+1,
            'verite':  CLASSES[label],
            'diagnostic_collectif': CLASSES[diagnostic] if diagnostic is not None else 'N/A',
            'confiance': round(conf, 4),
            'votes': [{**v, 'classe': CLASSES[v['pred']]} for v in votes]
        })

    # Résumé final
    print(f"\n{'='*55}")
    print(f"  RÉSULTATS COLLECTIFS")
    print(f"{'='*55}")
    print(f"  Précision collective  : {correct_collectif/nb_test*100:.0f}%")
    for vm_id in ['VM1', 'VM2', 'VM3']:
        nb_ok = correct_individuel.get(vm_id, 0)
        if vm_id in modeles:
            print(f"  Précision {vm_id}      : {nb_ok/nb_test*100:.0f}%")
    print(f"  Taux de consensus     : {nb_consensus/nb_test*100:.0f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/resultats_collectifs.json", "w") as f:
        json.dump(resultats_tous, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Résultats sauvegardés : results/resultats_collectifs.json")
    print(f"🚀 Lance : python thingsboard/mqtt_client.py")