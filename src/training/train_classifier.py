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

from src.dataset.oct_cc_dataset import OCTFrameDataset
from src.models.classifier import CCClassifier
from src.utils.io import load_annotation_excel

import argparse


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_patient_dirs(base_dir, patient_ids):
    base_dir = Path(base_dir)
    dirs = []
    for pid in patient_ids:
        hospital = "-".join(pid.split("-")[:2])
        patient_path = base_dir / hospital / pid
        if patient_path.exists():
            dirs.append(patient_path)
        else:
            print(f"Warning: {patient_path} not found, skipping.")
    return dirs


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
    patient_dirs = get_patient_dirs(config["data"]["base_dir"], patient_ids)

    # Patient-level split
    random.shuffle(patient_dirs)
    val_size = max(1, int(len(patient_dirs) * config["training"]["val_split"]))
    val_patient_dirs = patient_dirs[:val_size]
    train_patient_dirs = patient_dirs[val_size:]

    print(f"Train patients ({len(train_patient_dirs)}): {[p.name for p in train_patient_dirs]}")
    print(f"Val patients ({len(val_patient_dirs)}): {[p.name for p in val_patient_dirs]}")

    image_size = config["data"].get("image_size", 512)
    dicom_dir = Path(config["data"]["dicom_dir"])

    train_set = OCTFrameDataset(
        dicom_dir, train_patient_dirs,
        negative_frames_map=negative_frames_map,
        transform=get_transforms(train=True, image_size=image_size)
    )
    val_set = OCTFrameDataset(
        dicom_dir, val_patient_dirs,
        negative_frames_map=negative_frames_map,
        transform=get_transforms(train=False, image_size=image_size)
    )

    train_loader = DataLoader(train_set, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)

    model = CCClassifier(
        backbone=config["model"]["backbone"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
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
                "train_patients": [p.name for p in train_patient_dirs],
                "val_patients": [p.name for p in val_patient_dirs],
            }, output_dir / "best_classifier.pth")
            print("Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    main()