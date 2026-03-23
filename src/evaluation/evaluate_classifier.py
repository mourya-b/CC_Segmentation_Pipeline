import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.models.classifier import CCClassifier
from src.dataset.oct_cc_dataset import OCTFrameDataset
from src.utils.io import load_annotation_excel


def get_patient_dirs(base_dir, patient_ids):
    from pathlib import Path
    dirs = []
    for pid in patient_ids:
        hospital_code = pid.rsplit("-", 1)[0]
        p = Path(base_dir) / hospital_code / pid
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

    # Load annotations
    patient_ids, cc_frames_map, negative_frames_map = load_annotation_excel(
        cfg["data"]["annotation_excel"]
    )

    # Build dataset (same seed/split as training)
    patient_dirs = get_patient_dirs(cfg["data"]["base_dir"], patient_ids)
    dataset = OCTFrameDataset(
        dicom_dir=cfg["data"]["dicom_dir"],
        patient_dirs=patient_dirs,
        cc_frames_map=cc_frames_map,
        negative_frames_map=negative_frames_map,
    )

    # Reproduce val split
    torch.manual_seed(cfg["training"]["seed"])
    n = len(dataset)
    val_size = int(n * cfg["training"]["val_split"])
    indices = torch.randperm(n).tolist()
    val_indices = indices[:val_size]

    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CCClassifier(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

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
    print(f"Samples: {len(all_labels)} (pos: {all_labels.sum()}, neg: {(1-all_labels).sum()})")
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