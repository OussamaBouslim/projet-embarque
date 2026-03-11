"""
=============================================================
 dataset/preprocessing.py
 Dataset : Chest X-Ray Pneumonia (NORMAL vs PNEUMONIA)
=============================================================
"""
import os, shutil, random, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets

# ─── CONFIG ───
SEED          = 42
BATCH_SIZE    = 32
IMG_SIZE      = 224
DOSSIER_RAW   = "dataset/chest_xray"
DOSSIER_CLEAN = "dataset/chest_xray_clean"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def collecter_toutes_images():
    print("\n📂 Collecte de toutes les images...")
    toutes = {"NORMAL": [], "PNEUMONIA": []}
    for split in ["train", "val", "test"]:
        for classe in ["NORMAL", "PNEUMONIA"]:
            dossier = os.path.join(DOSSIER_RAW, split, classe)
            if os.path.exists(dossier):
                images = [
                    os.path.join(dossier, f)
                    for f in os.listdir(dossier)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                toutes[classe].extend(images)
    for c, imgs in toutes.items():
        print(f"  {c}: {len(imgs)} images")
    return toutes

def creer_dataset_propre(toutes_images):
    print("\n✂️  Division 70% / 15% / 15%...")
    if os.path.exists(DOSSIER_CLEAN):
        shutil.rmtree(DOSSIER_CLEAN)
    stats = {}
    for classe, images in toutes_images.items():
        random.shuffle(images)
        n       = len(images)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        n_test  = n - n_train - n_val
        splits  = {
            "train": images[:n_train],
            "val":   images[n_train:n_train+n_val],
            "test":  images[n_train+n_val:]
        }
        stats[classe] = {"train": n_train, "val": n_val, "test": n_test}
        for split_nom, fichiers in splits.items():
            dest = os.path.join(DOSSIER_CLEAN, split_nom, classe)
            os.makedirs(dest, exist_ok=True)
            for src in fichiers:
                shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
    print(f"\n  {'Classe':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for c, s in stats.items():
        emoji = "🟢" if c == "NORMAL" else "🔴"
        print(f"  {emoji} {c:<10} {s['train']:>8} {s['val']:>8} {s['test']:>8} {s['train']+s['val']+s['test']:>8}")
    return stats

def creer_transforms():
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])
    ])
    return transform_train, transform_test

def creer_dataloaders(transform_train, transform_test):
    print("\n🔄 Création des DataLoaders...")
    train_ds = datasets.ImageFolder(
        os.path.join(DOSSIER_CLEAN, "train"), transform=transform_train)
    val_ds   = datasets.ImageFolder(
        os.path.join(DOSSIER_CLEAN, "val"),   transform=transform_test)
    test_ds  = datasets.ImageFolder(
        os.path.join(DOSSIER_CLEAN, "test"),  transform=transform_test)

    # Équilibrage des classes (PNEUMONIA >> NORMAL)
    comptages    = [0] * len(train_ds.classes)
    for _, label in train_ds.samples:
        comptages[label] += 1
    poids_classes = [1.0 / c for c in comptages]
    poids_images  = [poids_classes[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(poids_images, len(poids_images), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False,  num_workers=0)

    print(f"  ✅ Train   : {len(train_ds):5d} images")
    print(f"  ✅ Val     : {len(val_ds):5d} images")
    print(f"  ✅ Test    : {len(test_ds):5d} images")
    print(f"  ✅ Classes : {train_ds.classes} → {train_ds.class_to_idx}")
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds

if __name__ == "__main__":
    debut = time.time()
    print("="*50)
    print("  PREPROCESSING — Chest X-Ray Pneumonia")
    print("="*50)

    toutes = collecter_toutes_images()
    stats  = creer_dataset_propre(toutes)
    t_train, t_test = creer_transforms()
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = \
        creer_dataloaders(t_train, t_test)

    # Test rapide
    imgs, labels = next(iter(train_loader))
    print(f"\n🧪 Test batch  : {imgs.shape}")
    print(f"   Labels      : {labels.tolist()[:10]}")
    print(f"   Min/Max px  : {imgs.min():.2f} / {imgs.max():.2f}")
    print(f"\n✅ Preprocessing OK en {time.time()-debut:.1f}s")
    print("🚀 Lance maintenant : python baseline/train.py")
