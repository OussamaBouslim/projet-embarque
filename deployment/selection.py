"""
=============================================================
 deployment/selection.py  — VERSION CORRIGEE
 Calcule le score pondéré et choisit la meilleure
 technique pour chaque VM
=============================================================
"""
import os, json
import pandas as pd
import numpy as np

def normaliser(serie):
    """Normalise entre 0 (pire) et 1 (meilleur)."""
    rng = serie.max() - serie.min()
    if rng == 0:
        return pd.Series([0.5]*len(serie), index=serie.index)
    return (serie - serie.min()) / rng

def score_vm(df_vm, vm_id):
    """Calcule le score pondéré selon les priorités de chaque VM."""
    ram_s  = 1 - normaliser(df_vm['ram_mo'])          # Moins RAM = mieux
    vit_s  = 1 - normaliser(df_vm['inference_ms'])    # Plus rapide = mieux
    prec_s = normaliser(df_vm['accuracy'])             # Plus précis = mieux

    if vm_id == 'VM1':
        score = 0.40*ram_s + 0.40*vit_s + 0.20*prec_s
    elif vm_id == 'VM2':
        score = 0.30*ram_s + 0.30*vit_s + 0.40*prec_s
    elif vm_id == 'VM3':
        score = 0.60*prec_s + 0.25*vit_s + 0.15*ram_s

    df_vm = df_vm.copy()
    df_vm['score'] = score
    return df_vm.sort_values('score', ascending=False)


if __name__ == "__main__":
    print("=" * 55)
    print("  SÉLECTION — Meilleure technique par VM")
    print("=" * 55)

    df = pd.read_csv("results/matrice_resultats.csv")

    # Afficher les colonnes disponibles pour debug
    print(f"\n  Colonnes CSV : {list(df.columns)}")
    print(f"  Total lignes : {len(df)}")

    # Garder seulement les résultats OK — colonne 'statut'
    df_ok = df[df['statut'] == 'OK'].copy()
    print(f"  Résultats OK : {len(df_ok)}")

    if df_ok.empty:
        print("\n⚠️  Aucun résultat OK trouvé !")
        print("   Vérifie que test_on_vm.py a bien tourné.")
        exit()

    selections = {}

    for vm_id in ['VM1', 'VM2', 'VM3']:
        df_vm = df_ok[df_ok['vm'] == vm_id].copy()

        if df_vm.empty:
            print(f"\n⚠️  {vm_id} : aucun résultat OK !")
            continue

        classement = score_vm(df_vm, vm_id)
        meilleure  = classement.iloc[0]
        selections[vm_id] = {
            "technique":    meilleure['technique'],
            "score":        round(float(meilleure['score']), 4),
            "accuracy":     meilleure['accuracy'],
            "ram_mo":       meilleure['ram_mo'],
            "inference_ms": meilleure['inference_ms'],
        }

        print(f"\n{'─'*52}")
        print(f"🖥️  {vm_id}")
        print(f"{'─'*52}")
        print(f"  Classement :")
        for _, row in classement.iterrows():
            print(f"    {row['technique']:5s}  score={row['score']:.3f} | "
                  f"acc={row['accuracy']:.1f}% | "
                  f"RAM={row['ram_mo']:.0f}Mo | "
                  f"{row['inference_ms']:.1f}ms")

        print(f"\n  ✅ MEILLEURE : {meilleure['technique']}")
        print(f"     Score     : {meilleure['score']:.4f}")
        print(f"     Accuracy  : {meilleure['accuracy']:.2f}%")
        print(f"     RAM       : {meilleure['ram_mo']:.0f} Mo")
        print(f"     Inférence : {meilleure['inference_ms']:.1f} ms")

    # Sauvegarder
    os.makedirs("results", exist_ok=True)
    with open("results/selections.json", "w") as f:
        json.dump(selections, f, indent=2)

    print(f"\n{'='*55}")
    print(f"✅ Sélections sauvegardées : results/selections.json")
    print(f"\n  Résumé :")
    for vm_id, sel in selections.items():
        print(f"  {vm_id} → {sel['technique']}  (score={sel['score']})")
    print(f"\n🚀 Lance : python collective/orchestrateur.py")