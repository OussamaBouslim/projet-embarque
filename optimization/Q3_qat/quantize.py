"""
=============================================================
 optimization/Q3_qat/quantize.py  — VERSION CORRIGEE WINDOWS
=============================================================
"""
import os, sys, json, warnings
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

# Fine-tuning avec simulation de quantification (lr très faible)
print("🏋️  Fine-tuning QAT simule (3 epochs)...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.00005)

model.train()
for epoch in range(3):
    total_loss = 0
    for i, (imgs, labels) in enumerate(train_loader):
        if i >= 50: break
        optimizer.zero_grad()
        # Simulation int8 : arrondir les poids pendant l'entraînement
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    scale = p.abs().max() / 127.0 + 1e-8
                    p.data = torch.round(p.data / scale) * scale
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / min(50, len(train_loader))
    _, acc_val, _, _, _ = evaluer(model, val_loader)
    print(f"  Epoch {epoch+1}/3 | Loss: {avg_loss:.4f} | Val Acc: {acc_val*100:.2f}%")

# Quantification dynamique du modele fine-tune
print("\n🔧 Application quantification dynamique...")
model.eval()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_q3 = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

os.makedirs("optimization/Q3_qat", exist_ok=True)
SAVE = "optimization/Q3_qat/model_q3.pt"
torch.save(model_q3.state_dict(), SAVE)

taille_q3   = os.path.getsize(SAVE) / (1024*1024)
taille_base = os.path.getsize("baseline/model_baseline.pt") / (1024*1024)
_, acc, f1, _, _ = evaluer(model_q3, test_loader)
temps_ms, _ = mesurer_inference(model_q3, test_loader)

res = {
    "technique":    "Q3 - QAT (fine-tuning simule Windows)",
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
print(f"   Inference   : {temps_ms} ms/image")
