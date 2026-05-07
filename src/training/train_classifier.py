import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.training.losses import FocalLoss
from src.dataset.oct_cc_dataset import OCTFrameDataset
from src.models.classifier import CCClassifier
from src.utils.io import load_annotation_excel

import argparse


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_patient_dirs(sources, patient_ids):
    """
    Search all source base_dirs for each patient and return
    list of (patient_dir, dicom_dir) tuples.
    """
    results = []
    for pid in patient_ids:
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
            print(f"Warning: {pid} not found in any source, skipping.")
    return results


def get_transforms(train=True, image_size=512):
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


def build_dataset(patient_dirs_with_dicoms, negative_frames_map, transform):
    """
    Build dataset handling multiple dicom_dirs per patient.
    Groups patients by dicom_dir and creates one dataset per group,
    then concatenates.
    """
    from torch.utils.data import ConcatDataset

    # Group by dicom_dir
    groups = {}
    for patient_dir, dicom_dir in patient_dirs_with_dicoms:
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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="Train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def get_criterion(config):
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "cross_entropy")
    if loss_type == "focal":
        return FocalLoss(
            alpha=loss_cfg.get("alpha", 0.75),
            gamma=loss_cfg.get("gamma", 2.0)
        )
    return nn.CrossEntropyLoss()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_classifier.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    excel_path = Path(config["data"]["annotation_excel"])
    patient_ids, cc_frames_map, negative_frames_map = load_annotation_excel(excel_path)

    sources = config["data"]["sources"]
    all_patient_dirs = get_patient_dirs(sources, patient_ids)

    print(f"Found {len(all_patient_dirs)} usable patients out of {len(patient_ids)}")

    # Patient-level split
    explicit_val = config["training"].get("val_patients", None)
    if explicit_val:
        val_patient_dirs = [(p, d) for p, d in all_patient_dirs if p.name in explicit_val]
        train_patient_dirs = [(p, d) for p, d in all_patient_dirs if p.name not in explicit_val]
        print(f"Using explicit val set from config")
    else:
        random.shuffle(all_patient_dirs)
        val_size = max(3, int(len(all_patient_dirs) * config["training"]["val_split"]))
        val_patient_dirs = all_patient_dirs[:val_size]
        train_patient_dirs = all_patient_dirs[val_size:]

    print(f"Train patients ({len(train_patient_dirs)}): {[p.name for p, _ in train_patient_dirs]}")
    print(f"Val patients ({len(val_patient_dirs)}): {[p.name for p, _ in val_patient_dirs]}")

    image_size = config["data"].get("image_size", 512)

    train_set = build_dataset(train_patient_dirs, negative_frames_map, get_transforms(True, image_size))
    val_set = build_dataset(val_patient_dirs, negative_frames_map, get_transforms(False, image_size))

    train_loader = DataLoader(train_set, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)

    model = CCClassifier(
        backbone=config["model"]["backbone"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"]
    ).to(device)

    criterion = get_criterion(config)
    print(f"Using loss: {config.get('loss', {}).get('type', 'cross_entropy')}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience = config["training"].get("early_stopping_patience", 10)
    epochs_no_improve = 0

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{config['training']['epochs']} "
              f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_patients": [p.name for p, _ in train_patient_dirs],
                "val_patients": [p.name for p, _ in val_patient_dirs],
            }, output_dir / "best_classifier.pth")
            print("Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    main()