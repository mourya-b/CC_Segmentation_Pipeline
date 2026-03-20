import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

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
    torch.manual_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    excel_path = Path(config["data"]["annotation_excel"])
    patient_ids, cc_frames_map, negative_frames_map = load_annotation_excel(excel_path)
    patient_dirs = get_patient_dirs(config["data"]["base_dir"], patient_ids)

    dicom_dir = Path(config["data"]["dicom_dir"])
    dataset = OCTFrameDataset(dicom_dir, patient_dirs, negative_frames_map=negative_frames_map)

    val_size = int(len(dataset) * config["training"]["val_split"])
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)

    model = CCClassifier(
        backbone=config["model"]["backbone"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{config['training']['epochs']} "
              f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_classifier.pth")
            print("Model saved.")


if __name__ == "__main__":
    main()