"""
=============================================================
 baseline/train.py
 Architecture : MobileNetV2 — Transfer Learning
 Dataset      : Chest X-Ray Pneumonia
=============================================================
"""
import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import psutil
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.preprocessing import creer_transforms, creer_dataloaders

# ─── CONFIG ───
NB_CLASSES = 2
NB_EPOCHS  = 10
LR         = 0.001
PATIENCE   = 3
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH  = "baseline/model_baseline.pt"

print(f"🖥️  Appareil : {DEVICE}")

def construire_modele():
    print("\n🏗️  Construction MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Geler toutes les couches
    for p in model.parameters():
        p.requires_grad = False
    # Remplacer le classificateur
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, NB_CLASSES)
    )
    # Dégeler les 3 derniers blocs
    for i, layer in enumerate(model.features):
        if i >= 15:
            for p in layer.parameters():
                p.requires_grad = True
    model = model.to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres entraînables : {trainable:,} / {total:,}")
    return model

def entrainer_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds_all.extend(out.argmax(1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average='weighted', zero_division=0)
    return total_loss / len(loader), acc, f1

def evaluer(model, loader):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item()
            preds_all.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average='weighted', zero_division=0)
    return total_loss / len(loader), acc, f1, preds_all, labels_all

def mesurer_inference(model, test_loader, nb=100):
    model.eval()
    temps, n = [], 0
    with torch.no_grad():
        for imgs, _ in test_loader:
            if n >= nb: break
            imgs = imgs.to(DEVICE)
            t = time.perf_counter()
            model(imgs)
            temps.append((time.perf_counter()-t) / imgs.shape[0] * 1000)
            n += imgs.shape[0]
    return round(np.mean(temps), 2), round(np.std(temps), 2)

if __name__ == "__main__":
    debut = time.time()
    print("="*50)
    print("  ENTRAÎNEMENT BASELINE — MobileNetV2")
    print("="*50)

    # Données
    t_train, t_test = creer_transforms()
    train_loader, val_loader, test_loader, _, _, _ = \
        creer_dataloaders(t_train, t_test)

    # Modèle
    model     = construire_modele()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5)

    # Entraînement
    historique = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
    best_loss, patience_c, best_epoch = float('inf'), 0, 0

    print(f"\n🚀 Entraînement ({NB_EPOCHS} epochs)...\n")
    for epoch in range(1, NB_EPOCHS+1):
        t_loss, t_acc, t_f1 = entrainer_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, v_f1, _, _ = evaluer(model, val_loader)
        scheduler.step(v_loss)

        historique['train_loss'].append(t_loss)
        historique['val_loss'].append(v_loss)
        historique['train_acc'].append(t_acc)
        historique['val_acc'].append(v_acc)

        print(f"Epoch {epoch:2d}/{NB_EPOCHS} | "
              f"Train Loss={t_loss:.4f} Acc={t_acc:.4f} | "
              f"Val Loss={v_loss:.4f} Acc={v_acc:.4f}")

        if v_loss < best_loss:
            best_loss, best_epoch, patience_c = v_loss, epoch, 0
            os.makedirs("baseline", exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  💾 Modèle sauvegardé (val_loss={v_loss:.4f})")
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                print(f"\n🛑 Early stopping à l'epoch {epoch}")
                break

    # Évaluation finale
    print("\n" + "="*50)
    print("  ÉVALUATION SUR LE JEU DE TEST")
    print("="*50)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    _, test_acc, test_f1, preds, labels = evaluer(model, test_loader)
    print(classification_report(labels, preds,
          target_names=['NORMAL','PNEUMONIA']))

    temps_ms, temps_std = mesurer_inference(model, test_loader)
    taille_mo = os.path.getsize(SAVE_PATH) / (1024*1024)
    ram_mo    = psutil.Process().memory_info().rss / (1024**2)

    print(f"\n{'═'*50}")
    print(f"  Accuracy  : {test_acc*100:.2f}%")
    print(f"  F1-Score  : {test_f1*100:.2f}%")
    print(f"  Taille    : {taille_mo:.2f} Mo")
    print(f"  RAM       : {ram_mo:.0f} Mo")
    print(f"  Inférence : {temps_ms} ± {temps_std} ms/image")
    print(f"  Durée     : {(time.time()-debut)/60:.1f} min")
    print(f"{'═'*50}")

    # Sauvegarder résultats JSON
    os.makedirs("results", exist_ok=True)
    res = {
        "modele": "MobileNetV2",
        "accuracy": round(test_acc, 4),
        "f1_score": round(test_f1, 4),
        "taille_mo": round(taille_mo, 2),
        "ram_mo": round(ram_mo, 1),
        "inference_ms": temps_ms,
        "inference_std": temps_std,
        "meilleure_epoch": best_epoch,
        "historique": historique
    }
    with open("results/baseline_results.json", "w") as f:
        json.dump(res, f, indent=2)

    # Courbes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0f172a')
    epochs = range(1, len(historique['train_loss'])+1)
    for ax, (t_val, v_val, titre) in zip(axes, [
        (historique['train_loss'], historique['val_loss'], 'Loss'),
        (historique['train_acc'],  historique['val_acc'],  'Accuracy')]):
        ax.set_facecolor('#1e293b')
        ax.plot(epochs, t_val, 'o-', color='#3b82f6', label='Train')
        ax.plot(epochs, v_val, 's-', color='#f97316', label='Val')
        ax.set_title(titre, color='white')
        ax.legend(facecolor='#0f172a', labelcolor='white')
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#334155')
    plt.tight_layout()
    plt.savefig("results/baseline_courbes.png", dpi=100,
                bbox_inches='tight', facecolor='#0f172a')
    plt.close()

    print(f"\n✅ Fichiers créés :")
    print(f"   baseline/model_baseline.pt")
    print(f"   results/baseline_results.json")
    print(f"   results/baseline_courbes.png")
    print(f"\n🚀 Lance : python optimization/Q1_dynamic_quant/quantize.py")
