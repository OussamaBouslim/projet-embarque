"""
=============================================================
 deployment/selection.py
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
    if rng == 0: return pd.Series([0.5]*len(serie), index=serie.index)
    return (serie - serie.min()) / rng

def score_vm(df_vm, vm_id):
    """Calcule le score pondéré selon les priorités de chaque VM."""
    ram_s  = 1 - normaliser(df_vm['ram_mo'])     # Moins RAM = mieux
    cpu_s  = 1 - normaliser(df_vm['cpu_pct'])    # Moins CPU = mieux
    vit_s  = 1 - normaliser(df_vm['temps_ms'])   # Plus rapide = mieux
    prec_s = normaliser(df_vm['accuracy'])        # Plus précis = mieux

    if vm_id == 'VM1':
        # Très contrainte → RAM et CPU prioritaires
        score = 0.40*ram_s + 0.40*cpu_s + 0.20*prec_s
    elif vm_id == 'VM2':
        # Intermédiaire → équilibre
        score = 0.30*ram_s + 0.30*vit_s + 0.40*prec_s
    elif vm_id == 'VM3':
        # Capable → précision prioritaire
        score = 0.60*prec_s + 0.25*vit_s + 0.15*ram_s

    df_vm = df_vm.copy()
    df_vm['score'] = score
    return df_vm.sort_values('score', ascending=False)


if __name__ == "__main__":
    print("="*55)
    print("  SÉLECTION — Meilleure technique par VM")
    print("="*55)

    df = pd.read_csv("results/matrice_resultats.csv")

    # Garder seulement les résultats OK (pas OOM)
    df_ok = df[df['status'] == 'OK'].copy()

    selections = {}
    for vm_id in ['VM1', 'VM2', 'VM3']:
        df_vm = df_ok[df_ok['vm'] == vm_id].copy()
        if df_vm.empty:
            print(f"\n⚠️  {vm_id} : aucun résultat OK !")
            continue

        classement = score_vm(df_vm, vm_id)
        meilleure  = classement.iloc[0]
        selections[vm_id] = meilleure['technique']

        print(f"\n{'─'*50}")
        print(f"🖥️  {vm_id}")
        print(f"{'─'*50}")
        print(f"  Classement :")
        for _, row in classement.iterrows():
            print(f"    {row['technique']:30s} score={row['score']:.4f} | "
                  f"acc={row['accuracy']*100:.1f}% | "
                  f"RAM={row['ram_mo']:.0f}Mo | "
                  f"{row['temps_ms']:.0f}ms")
        print(f"\n  ✅ MEILLEURE : {meilleure['technique']}")
        print(f"     Score     : {meilleure['score']:.4f}")
        print(f"     Accuracy  : {meilleure['accuracy']*100:.2f}%")
        print(f"     RAM       : {meilleure['ram_mo']:.0f} Mo")
        print(f"     Inférence : {meilleure['temps_ms']:.0f} ms")

    # Sauvegarder
    os.makedirs("results", exist_ok=True)
    with open("results/selections.json", "w") as f:
        json.dump(selections, f, indent=2)

    print(f"\n✅ Sélections sauvegardées : results/selections.json")
    print("🚀 Lance : python collective/orchestrateur.py")
