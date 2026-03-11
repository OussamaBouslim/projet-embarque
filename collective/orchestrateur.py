"""
=============================================================
 collective/orchestrateur.py
 Intelligence collective : vote pondéré des 3 VM
=============================================================
"""
import os, sys, json, time
import torch
import torch.nn.functional as F
import psutil
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.train import construire_modele, evaluer
from dataset.preprocessing import creer_transforms, creer_dataloaders

DEVICE  = torch.device('cpu')
CLASSES = ['NORMAL', 'PNEUMONIA']

# ── Précisions historiques (remplace par tes vrais chiffres de Phase 3) ──
PRECISIONS_VM = {'VM1': 0.87, 'VM2': 0.91, 'VM3': 0.94}

# ── Chemins des meilleurs modèles par VM (résultats Phase 4) ──
MODELES_VM = {
    'VM1': 'optimization/Q1_dynamic_quant/model_q1.pt',
    'VM2': 'optimization/Q2_static_ptq/model_q2.pt',
    'VM3': 'optimization/P2_structured/model_p2.pt',
}


def charger_modeles():
    """Charge les 3 modèles en mémoire."""
    modeles = {}
    for vm_id, chemin in MODELES_VM.items():
        if os.path.exists(chemin):
            m = construire_modele()
            m.load_state_dict(torch.load(chemin, map_location=DEVICE))
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
        # Vérifier surcharge
        cpu = psutil.cpu_percent(interval=0.2)
        ram = psutil.virtual_memory().percent
        if cpu > 85 or ram > 90:
            print(f"  ⚠️  {vm_id} surchargée (CPU={cpu}% RAM={ram}%) → ignorée")
            continue

        pred, conf, probas = inferer(model, image)
        resultats.append({
            'vm': vm_id, 'pred': pred,
            'confiance': conf, 'probas': probas
        })
        print(f"  [{vm_id}] → {CLASSES[pred]} ({conf:.0%})")

    if not resultats:
        return None, 0, []

    diagnostic, conf_collective = vote_pondere(resultats)

    # Validation : si confiance < 70% → alerter
    if conf_collective < 0.70:
        print(f"  ⚠️  Confiance faible ({conf_collective:.0%}) → Validation renforcée !")

    # Taux de consensus (les 3 VM sont-elles d'accord ?)
    predictions = [r['pred'] for r in resultats]
    consensus   = all(p == predictions[0] for p in predictions)

    print(f"\n  🏥 Diagnostic final : {CLASSES[diagnostic]}")
    print(f"  📊 Confiance collective : {conf_collective:.0%}")
    print(f"  🤝 Consensus : {'✅ OUI' if consensus else '❌ NON'}")
    return diagnostic, conf_collective, resultats


if __name__ == "__main__":
    print("="*55)
    print("  INTELLIGENCE COLLECTIVE — Test sur 10 exemples")
    print("="*55)

    # Charger modèles et données
    modeles = charger_modeles()
    t_train, t_test = creer_transforms()
    _, _, test_loader, _, _, test_ds = creer_dataloaders(t_train, t_test)

    # Évaluer sur 10 exemples
    correct_collectif = 0
    correct_individuel = {'VM1': 0, 'VM2': 0, 'VM3': 0}
    nb_consensus   = 0
    nb_test        = 10

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
    for vm_id, nb_ok in correct_individuel.items():
        print(f"  Précision {vm_id}      : {nb_ok/nb_test*100:.0f}%")
    print(f"  Taux de consensus     : {nb_consensus/nb_test*100:.0f}%")

    # Sauvegarder
    os.makedirs("results", exist_ok=True)
    with open("results/resultats_collectifs.json", "w") as f:
        json.dump(resultats_tous, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Résultats sauvegardés : results/resultats_collectifs.json")
