import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import argparse
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.training.losses import FocalLoss
from src.dataset.oct_cc_dataset import OCTFrameDataset
from src.models.classifier import CCClassifier
from src.utils.io import load_annotation_excel
from src.training.train_classifier import (
    get_patient_dirs, get_transforms, build_dataset,
    freeze_backbone, unfreeze_last_blocks, get_optimizer,
    train_one_epoch, evaluate, get_criterion
)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_positive_frame_count(patient_dir, cc_frames_map):
    """Get number of positive CC frames for a patient."""
    pid = patient_dir.name
    frames = cc_frames_map.get(pid, [])
    return len(frames)


def stratified_kfold_patients(patient_dirs, cc_frames_map, n_folds=5, seed=42):
    """
    Split patients into n_folds stratified by positive frame count.
    Returns list of (train_indices, val_indices) tuples.
    """
    random.seed(seed)

    # Get positive frame counts and sort into bins
    counts = [(i, get_positive_frame_count(p, cc_frames_map)) for i, (p, d) in enumerate(patient_dirs)]

    # Bin patients: high (>15), medium (5-15), low (1-5), zero (0)
    bins = defaultdict(list)
    for idx, count in counts:
        if count == 0:
            bins['zero'].append(idx)
        elif count <= 5:
            bins['low'].append(idx)
        elif count <= 15:
            bins['medium'].append(idx)
        else:
            bins['high'].append(idx)

    print(f"\nStratification bins:")
    for bin_name, idxs in bins.items():
        print(f"  {bin_name}: {len(idxs)} patients — {[patient_dirs[i][0].name for i in idxs]}")

    # Shuffle each bin
    for bin_name in bins:
        random.shuffle(bins[bin_name])

    # Assign patients to folds round-robin within each bin
    folds = [[] for _ in range(n_folds)]
    for bin_name, idxs in bins.items():
        for i, idx in enumerate(idxs):
            folds[i % n_folds].append(idx)

    # Build train/val splits
    splits = []
    for fold_idx in range(n_folds):
        val_indices = folds[fold_idx]
        train_indices = [idx for f in range(n_folds) if f != fold_idx for idx in folds[f]]
        splits.append((train_indices, val_indices))

    return splits


def evaluate_fold(model, loader, device):
    """Run eval and return raw predictions and labels."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def train_fold(config, train_patient_dirs, val_patient_dirs, negative_frames_map,
               device, fold_idx, output_dir):
    """Train one fold and return best checkpoint path."""
    image_size = config["data"].get("image_size", 512)

    train_set = build_dataset(train_patient_dirs, negative_frames_map, get_transforms(True, image_size))
    val_set = build_dataset(val_patient_dirs, negative_frames_map, get_transforms(False, image_size))

    train_loader = DataLoader(train_set, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config["training"]["batch_size"],
                            shuffle=False, num_workers=0)

    model = CCClassifier(
        backbone=config["model"]["backbone"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"]
    ).to(device)

    criterion = get_criterion(config)

    freeze_epochs = config["training"].get("freeze_epochs", 15)
    unfreeze_blocks = config["training"].get("unfreeze_blocks", 7)
    backbone_lr_scale = config["training"].get("backbone_lr_scale", 0.01)
    total_epochs = config["training"]["epochs"]
    patience = config["training"].get("early_stopping_patience", 10)

    # Phase 1
    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    phase = 1
    checkpoint_path = output_dir / f"fold_{fold_idx}_best.pth"

    for epoch in range(total_epochs):
        if epoch == freeze_epochs and phase == 1:
            print(f"  [Fold {fold_idx}] Epoch {epoch+1}: Switching to Phase 2")
            unfreeze_last_blocks(model, num_blocks=unfreeze_blocks)
            optimizer = get_optimizer(model, config, backbone_lr_scale)
            t_max = total_epochs - freeze_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-7)
            phase = 2

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if phase == 1:
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(f"  [Fold {fold_idx}] Epoch {epoch+1}/{total_epochs} [Phase {phase}] "
              f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                        "val_loss": val_loss}, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [Fold {fold_idx}] Early stopping at epoch {epoch+1}")
                break

    return checkpoint_path, val_loader, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_classifier_cluster.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    excel_path = Path(config["data"]["annotation_excel"])
    patient_ids, cc_frames_map, negative_frames_map = load_annotation_excel(excel_path)

    sources = config["data"]["sources"]
    all_patient_dirs = get_patient_dirs(sources, patient_ids)
    print(f"Found {len(all_patient_dirs)} usable patients")

    # CV config
    cv_cfg = config.get("cv", {})
    n_folds = cv_cfg.get("n_folds", 5)
    pin_to_train = cv_cfg.get("pin_to_train", ["NLD-TERG-0002", "NLD-UMCG-0001-LAD"])

    # Separate pinned patients
    pinned = [(p, d) for p, d in all_patient_dirs if p.name in pin_to_train]
    foldable = [(p, d) for p, d in all_patient_dirs if p.name not in pin_to_train]

    print(f"Pinned to train always: {[p.name for p, _ in pinned]}")
    print(f"Patients available for CV folds: {len(foldable)}")

    # Get stratified splits
    splits = stratified_kfold_patients(foldable, cc_frames_map, n_folds=n_folds, seed=seed)

    # Output dir
    output_dir = Path(config["output_dir"]) / "cv"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run CV
    fold_metrics = []

    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")

        train_patient_dirs = pinned + [foldable[i] for i in train_indices]
        val_patient_dirs = [foldable[i] for i in val_indices]

        print(f"  Train: {len(train_patient_dirs)} patients")
        print(f"  Val: {[foldable[i][0].name for i in val_indices]}")

        # Train fold
        checkpoint_path, val_loader, model = train_fold(
            config, train_patient_dirs, val_patient_dirs,
            negative_frames_map, device, fold_idx + 1, output_dir
        )

        # Load best checkpoint for this fold
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        # Evaluate
        preds, probs, labels = evaluate_fold(model, val_loader, device)

        if len(np.unique(labels)) < 2:
            print(f"  [Fold {fold_idx+1}] Warning: only one class in val set, skipping AUC")
            auc = float('nan')
        else:
            auc = roc_auc_score(labels, probs)

        metrics = {
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "auc": auc,
            "val_loss": ckpt["val_loss"],
            "best_epoch": ckpt["epoch"],
            "n_val_patients": len(val_indices),
            "n_val_samples": len(labels),
            "n_pos": int(labels.sum()),
            "n_neg": int((1 - labels).sum()),
        }
        fold_metrics.append(metrics)

        print(f"\n  [Fold {fold_idx+1}] Results:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")
        print(f"    AUC:       {metrics['auc']:.4f}")
        print(f"    Val Loss:  {metrics['val_loss']:.4f}")

    # Aggregate results
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for metric in ["precision", "recall", "f1", "auc"]:
        values = [m[metric] for m in fold_metrics if not np.isnan(m[metric])]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric.capitalize():12}: {mean:.4f} ± {std:.4f}")

    print(f"\nPer-fold breakdown:")
    print(f"{'Fold':>6} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8} {'ValLoss':>10} {'Epoch':>7} {'Samples':>8}")
    for i, m in enumerate(fold_metrics):
        print(f"{i+1:>6} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
              f"{m['auc']:>8.4f} {m['val_loss']:>10.4f} {m['best_epoch']:>7} {m['n_val_samples']:>8}")

    # Save results
    import json
    results_path = output_dir / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump(fold_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()