import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score

from src.dataset.oct_cc_dataset import OCTFrameDataset
from src.models.classifier import CCClassifier
from src.utils.io import load_annotation_excel
import argparse


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    def expand(obj):
        if isinstance(obj, dict): return {k: expand(v) for k, v in obj.items()}
        if isinstance(obj, list): return [expand(v) for v in obj]
        if isinstance(obj, str): return os.path.expandvars(obj)
        return obj
    return expand(config)


def get_patient_dirs(sources, patient_ids):
    results = []
    for pid in patient_ids:
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
            print(f"Warning: {pid} not found in any source, skipping.")
    return results


def get_transforms(train=True, image_size=512):
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.ElasticTransform(alpha=1, sigma=10, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


def build_dataset(patient_dirs_with_dicoms, negative_frames_map, transform, mask_cfg):
    from torch.utils.data import ConcatDataset
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
            mask_inner_frac=mask_cfg.get("inner_frac", 0.08),
            mask_outer_frac=mask_cfg.get("outer_frac", 0.45),
        )
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def freeze_backbone(model):
    head_names = model.get_head_param_names()
    for name, param in model.named_parameters():
        if any(h in name for h in head_names):
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone frozen — training heads only. Trainable params: {trainable:,}")


def unfreeze_last_blocks(model, num_blocks=3):
    for param in model.parameters():
        param.requires_grad = False
    head_names = model.get_head_param_names()
    for name, param in model.named_parameters():
        if any(h in name for h in head_names):
            param.requires_grad = True
    layers = model.get_backbone_layers()
    for layer in layers[-num_blocks:]:
        for param in layer.parameters():
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfroze last {num_blocks}/{len(layers)} blocks + heads. Trainable: {trainable:,}")


def get_optimizer(model, config, backbone_lr_scale=0.1):
    base_lr = config["training"]["learning_rate"]
    wd = config["training"].get("weight_decay", 1e-4)
    head_names = model.get_head_param_names()

    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and any(h in n for h in head_names)]
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(h in n for h in head_names)]

    param_groups = [
        {"params": head_params, "lr": base_lr},
        {"params": backbone_params, "lr": base_lr * backbone_lr_scale},
    ]
    return torch.optim.Adam(param_groups, weight_decay=wd)


def compute_losses(cls_logits, seg_logits, masks, labels, seg_weight, use_aux_seg):
    cls_loss = F.binary_cross_entropy_with_logits(cls_logits, labels)

    if use_aux_seg and seg_logits is not None:
        # Downsample mask to seg_logits resolution
        target = F.interpolate(
            masks.unsqueeze(1),                 # (B, 1, H, W)
            size=seg_logits.shape[-2:],         # e.g. (16, 16)
            mode="area",
        )
        target = (target > 0).float()
        seg_loss = F.binary_cross_entropy_with_logits(seg_logits, target)
    else:
        seg_loss = torch.tensor(0.0, device=cls_logits.device)

    total = cls_loss + seg_weight * seg_loss
    return total, cls_loss.detach(), seg_loss.detach()


def train_one_epoch(model, loader, optimizer, device, seg_weight, use_aux_seg):
    model.train()
    total_loss = total_cls = total_seg = 0.0
    correct = total = 0

    for images, masks, labels in tqdm(loader, desc="Train"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        cls_logits, seg_logits = model(images)
        loss, cls_l, seg_l = compute_losses(cls_logits, seg_logits, masks, labels,
                                            seg_weight, use_aux_seg)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls += cls_l.item()
        total_seg += seg_l.item()
        preds = (torch.sigmoid(cls_logits) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    n = len(loader)
    return total_loss / n, total_cls / n, total_seg / n, correct / total


def evaluate(model, loader, device, seg_weight, use_aux_seg):
    model.eval()
    total_loss = total_cls = total_seg = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, masks, labels in tqdm(loader, desc="Val"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            cls_logits, seg_logits = model(images)
            loss, cls_l, seg_l = compute_losses(cls_logits, seg_logits, masks, labels,
                                                seg_weight, use_aux_seg)
            total_loss += loss.item()
            total_cls += cls_l.item()
            total_seg += seg_l.item()

            probs = torch.sigmoid(cls_logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    n = len(loader)
    return total_loss / n, total_cls / n, total_seg / n, auc, probs, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_classifier.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    excel_path = Path(config["data"]["annotation_excel"])
    patient_ids, cc_frames_map, negative_frames_map = load_annotation_excel(excel_path)

    sources = config["data"]["sources"]
    all_patient_dirs = get_patient_dirs(sources, patient_ids)
    print(f"Found {len(all_patient_dirs)} usable patients out of {len(patient_ids)}")

    explicit_val = config["training"].get("val_patients", None)
    if explicit_val:
        val_patient_dirs = [(p, d) for p, d in all_patient_dirs if p.name in explicit_val]
        train_patient_dirs = [(p, d) for p, d in all_patient_dirs if p.name not in explicit_val]
        print("Using explicit val set from config")
    else:
        random.shuffle(all_patient_dirs)
        val_size = max(3, int(len(all_patient_dirs) * config["training"]["val_split"]))
        val_patient_dirs = all_patient_dirs[:val_size]
        train_patient_dirs = all_patient_dirs[val_size:]

    print(f"Train patients ({len(train_patient_dirs)}): {[p.name for p, _ in train_patient_dirs]}")
    print(f"Val patients ({len(val_patient_dirs)}): {[p.name for p, _ in val_patient_dirs]}")

    image_size = config["data"].get("image_size", 512)
    mask_cfg = config.get("mask", {"inner_frac": 0.08, "outer_frac": 0.45})

    train_set = build_dataset(train_patient_dirs, negative_frames_map,
                              get_transforms(True, image_size), mask_cfg)
    val_set = build_dataset(val_patient_dirs, negative_frames_map,
                            get_transforms(False, image_size), mask_cfg)

    train_loader = DataLoader(train_set, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=config["training"].get("num_workers", 0),
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config["training"]["batch_size"],
                            shuffle=False, num_workers=config["training"].get("num_workers", 0),
                            pin_memory=True)

    use_aux_seg = config.get("loss", {}).get("use_aux_seg", True)
    seg_weight = config.get("loss", {}).get("seg_weight", 0.3)
    print(f"Aux seg head: {use_aux_seg}, seg_weight: {seg_weight}")

    model = CCClassifier(
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        use_aux_seg=use_aux_seg,
    ).to(device)

    freeze_epochs = config["training"].get("freeze_epochs", 10)
    unfreeze_blocks = config["training"].get("unfreeze_blocks", 3)
    backbone_lr_scale = config["training"].get("backbone_lr_scale", 0.1)
    total_epochs = config["training"]["epochs"]

    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5  # mode='max' since we track AUC
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = -1.0
    patience = config["training"].get("early_stopping_patience", 11)
    epochs_no_improve = 0
    phase = 1

    for epoch in range(total_epochs):
        if epoch == freeze_epochs and phase == 1:
            print(f"\n--- Epoch {epoch+1}: Phase 2 — unfreezing {unfreeze_blocks} blocks ---")
            unfreeze_last_blocks(model, num_blocks=unfreeze_blocks)
            optimizer = get_optimizer(model, config, backbone_lr_scale)
            t_max = total_epochs - freeze_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=1e-7
            )
            epochs_no_improve = 0
            phase = 2

        tr_loss, tr_cls, tr_seg, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, seg_weight, use_aux_seg
        )
        vl_loss, vl_cls, vl_seg, vl_auc, _, _ = evaluate(
            model, val_loader, device, seg_weight, use_aux_seg
        )

        if phase == 1:
            scheduler.step(vl_auc)
        else:
            scheduler.step()

        print(f"Epoch {epoch+1}/{total_epochs} [Phase {phase}] "
              f"| Train loss {tr_loss:.4f} (cls {tr_cls:.4f}, seg {tr_seg:.4f}) acc {tr_acc:.4f} "
              f"| Val loss {vl_loss:.4f} (cls {vl_cls:.4f}, seg {vl_seg:.4f}) AUC {vl_auc:.4f}")

        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": vl_auc,
                "val_loss": vl_loss,
                "train_patients": [p.name for p, _ in train_patient_dirs],
                "val_patients": [p.name for p, _ in val_patient_dirs],
                "config": config,
            }, output_dir / "best_classifier.pth")
            print(f"Model saved. Best val AUC: {best_val_auc:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


if __name__ == "__main__":
    main()