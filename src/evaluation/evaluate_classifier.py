import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from src.models.classifier import CCClassifier
from src.dataset.oct_cc_dataset import OCTFrameDataset
from src.utils.io import load_annotation_excel


def get_val_patient_dirs(sources, val_patient_names):
    """Find val patient dirs and their corresponding dicom_dir across all sources."""
    results = []
    for pid in val_patient_names:
        pid = str(pid).strip()
        if not pid or pid == 'nan':
            continue
        hospital = "-".join(pid.split("-")[:2])
        found = False
        for source in sources:
            base_dir = Path(source["base_dir"])
            dicom_dir = Path(source["dicom_dir"])
            patient_path = base_dir / hospital / pid
            dcm_path = dicom_dir / f"{pid}.dcm"
            if patient_path.exists() and dcm_path.exists():
                results.append((patient_path, dicom_dir))
                found = True
                break
        if not found:
            print(f"Warning: val patient {pid} not found in any source")
    return results


def build_val_dataset(val_patient_dirs_with_dicoms, negative_frames_map, transform):
    """Build val dataset handling multiple dicom sources."""
    groups = {}
    for patient_dir, dicom_dir in val_patient_dirs_with_dicoms:
        key = str(dicom_dir)
        if key not in groups:
            groups[key] = {"dicom_dir": dicom_dir, "patient_dirs": []}
        groups[key]["patient_dirs"].append(patient_dir)

    datasets = []
    for key, group in groups.items():
        ds = OCTFrameDataset(
            dicom_dir=group["dicom_dir"],
            patient_dirs=group["patient_dirs"],
            negative_frames_map=negative_frames_map,
            transform=transform,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


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

    val_patient_names = checkpoint.get("val_patients", None)
    if val_patient_names is None:
        print("ERROR: checkpoint does not contain val_patients.")
        return

    print(f"Val patients from checkpoint: {val_patient_names}")

    # Load annotations
    patient_ids, cc_frames_map, negative_frames_map = load_annotation_excel(
        cfg["data"]["annotation_excel"]
    )

    # Find val patient dirs across all sources
    sources = cfg["data"]["sources"]
    val_patient_dirs = get_val_patient_dirs(sources, val_patient_names)

    # Build val transform
    image_size = cfg["data"].get("image_size", 512)
    val_transform = Compose([
        Resize(image_size, image_size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_set = build_val_dataset(val_patient_dirs, negative_frames_map, val_transform)

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
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