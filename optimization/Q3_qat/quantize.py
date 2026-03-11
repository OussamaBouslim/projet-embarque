"""
=============================================================
 optimization/Q3_qat/quantize.py
 Technique : Quantification Aware Training (QAT)
=============================================================
"""
import os, sys, json
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline.train import construire_modele, evaluer, mesurer_inference
from dataset.preprocessing import creer_transforms, creer_dataloaders

DEVICE = torch.device('cpu')

print("📦 Chargement du modèle baseline...")
model = construire_modele()
model.load_state_dict(torch.load("baseline/model_baseline.pt", map_location=DEVICE))

t_train, t_test = creer_transforms()
train_loader, val_loader, test_loader, _, _, _ = creer_dataloaders(t_train, t_test)

# ── QAT : simuler la quantification PENDANT l'entraînement ──
print("🔧 Préparation QAT...")
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

# Fine-tuning avec quantification simulée (3 epochs suffisent)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

print("🏋️  Fine-tuning QAT (3 epochs)...")
for epoch in range(3):
    model.train()
    total_loss = 0
    for i, (imgs, labels) in enumerate(train_loader):
        if i >= 50: break  # Limiter pour aller plus vite
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/3 | Loss: {total_loss/min(50,len(train_loader)):.4f}")

# Convertir en modèle quantifié final
model.eval()
torch.quantization.convert(model, inplace=True)

os.makedirs("optimization/Q3_qat", exist_ok=True)
SAVE = "optimization/Q3_qat/model_q3.pt"
torch.save(model.state_dict(), SAVE)

taille_q3   = os.path.getsize(SAVE) / (1024*1024)
taille_base = os.path.getsize("baseline/model_baseline.pt") / (1024*1024)
_, acc, f1, _, _ = evaluer(model, test_loader)
temps_ms, _ = mesurer_inference(model, test_loader)

res = {
    "technique":    "Q3 - QAT",
    "accuracy":     round(acc, 4),
    "f1_score":     round(f1, 4),
    "taille_mo":    round(taille_q3, 2),
    "compression":  round(taille_base / taille_q3, 2),
    "inference_ms": temps_ms
}
with open("optimization/Q3_qat/results_q3.json", "w") as f:
    json.dump(res, f, indent=2)

print(f"\n✅ [Q3] Résultats :")
print(f"   Accuracy    : {acc*100:.2f}%")
print(f"   Taille      : {taille_q3:.2f} Mo")
print(f"   Compression : {taille_base/taille_q3:.2f}x")
print(f"   Inférence   : {temps_ms} ms/image")
