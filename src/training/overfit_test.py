# src/training/overfit_test.py
import sys
from pathlib import Path
sys.path.insert(0, '/data/diag/mouryaBandaru/CC_Segmentation_Pipeline')

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.classifier import CCClassifier
from src.training.losses import FocalLoss
from src.dataset.oct_cc_dataset import OCTFrameDataset
from src.utils.io import load_annotation_excel
from pathlib import Path

# --- CONFIG ---
EXCEL = '/data/diag/mouryaBandaru/data/CC_Annotations_v3.xlsx'
DICOM_DIR = Path('/data/diag/rubenvdw/Dataset/DICOMS_Pectus')
SEG_DIR = Path('/data/diag/mouryaBandaru/data/PECTUS_segmentations')
PATIENT = 'NLD-TERG-0002'  # high CC count patient
EPOCHS = 200
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")

# Load annotations
patient_ids, cc_frames_map, negative_frames_map = load_annotation_excel(EXCEL)

# Build minimal dataset — no augmentation
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

patient_dir = SEG_DIR / 'NLD-TERG' / PATIENT
dataset = OCTFrameDataset(
    dicom_dir=DICOM_DIR,
    patient_dirs=[patient_dir],
    negative_frames_map=negative_frames_map,
    transform=transform,
)

print(f"Dataset size: {len(dataset)}")
print(f"Samples: {[(s[0], s[3]) for s in dataset.samples[:10]]}")  # frame_idx, label

# Take 3 pos + 3 neg samples only
pos_indices = [i for i, s in enumerate(dataset.samples) if s[3] == 1][:3]
neg_indices = [i for i, s in enumerate(dataset.samples) if s[3] == 0][:3]
subset_indices = pos_indices + neg_indices
subset = Subset(dataset, subset_indices)

print(f"\nOverfit subset:")
for i in subset_indices:
    s = dataset.samples[i]
    print(f"  frame {s[0]}, label {s[3]}")

loader = DataLoader(subset, batch_size=6, shuffle=True)

# Model
model = CCClassifier(backbone='efficientnet_b0', num_classes=2, pretrained=True).to(DEVICE)
criterion = FocalLoss(alpha=0.75, gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Unfreeze everything for overfit test
for param in model.parameters():
    param.requires_grad = True

print(f"\nTraining for {EPOCHS} epochs on 6 samples...")
print(f"{'Epoch':>6} {'Loss':>10} {'Acc':>8} {'AUC':>8} {'alpha_t':>12} {'pt':>10}")

for epoch in range(EPOCHS):
    model.train()
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)

        # Debug focal loss components
        import torch.nn.functional as F
        ce_loss = F.cross_entropy(outputs, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = 0.75 * labels.float() + (1 - 0.75) * (1 - labels.float())
        focal = alpha_t * (1 - pt) ** 2.0 * ce_loss

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        all_preds, all_probs, all_labels = [], [], []
        for images, labels in loader:
            images = images.to(DEVICE)
            out = model(images)
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = float('nan')

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            print(f"{epoch+1:>6} {loss.item():>10.4f} {acc:>8.4f} {auc:>8.4f} "
                  f"{alpha_t.mean().item():>12.4f} {pt.mean().item():>10.4f}")

print("\nDone")