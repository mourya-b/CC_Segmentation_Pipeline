import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.models.classifier import CCClassifier
from src.dataset.oct_cc_dataset import OCTFrameDataset
from src.utils.io import load_annotation_excel


def get_patient_dirs(base_dir, patient_ids):
    dirs = []
    base_dir = Path(base_dir)
    for pid in patient_ids:
        hospital = "-".join(pid.split("-")[:2])
        p = base_dir / hospital / pid
        if p.exists():
            dirs.append(p)
    return dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_classifier_cluster.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to best_classifier.pth")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load checkpoint first to get val patients
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load val patients from checkpoint
    val_patient_names = checkpoint.get("val_patients", None)
    if val_patient_names is None:
        print("ERROR: checkpoint does not contain val_patients. Cannot reproduce val split.")
        return

    print(f"Val patients from checkpoint: {val_patient_names}")

    # Load annotations
    patient_ids, cc_frames_map, negative_frames_map = load_annotation_excel(
        cfg["data"]["annotation_excel"]
    )

    # Build val patient dirs from checkpoint val_patients
    base_dir = Path(cfg["data"]["base_dir"])
    val_patient_dirs = []
    for pid in val_patient_names:
        hospital = "-".join(pid.split("-")[:2])
        p = base_dir / hospital / pid
        if p.exists():
            val_patient_dirs.append(p)
        else:
            print(f"Warning: val patient dir not found: {p}")

    # Build val dataset using only val patients
    from albumentations import Compose, Resize, Normalize
    from albumentations.pytorch import ToTensorV2
    val_transform = Compose([
        Resize(cfg["data"].get("image_size", 512), cfg["data"].get("image_size", 512)),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_set = OCTFrameDataset(
        dicom_dir=cfg["data"]["dicom_dir"],
        patient_dirs=val_patient_dirs,
        negative_frames_map=negative_frames_map,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    # Load model
    model = CCClassifier(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Evaluating checkpoint from epoch {checkpoint.get('epoch', '?')} "
          f"| Val loss: {checkpoint.get('val_loss', '?'):.4f}")

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    print("\n=== Evaluation Results ===")
    print(f"Samples: {len(all_labels)} (pos: {int(all_labels.sum())}, neg: {int((1-all_labels).sum())})")
    print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds):.4f}")
    print(f"F1:        {f1_score(all_labels, all_preds):.4f}")
    print(f"AUC:       {roc_auc_score(all_labels, all_probs):.4f}")
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(all_labels, all_preds))
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))


if __name__ == "__main__":
    main()