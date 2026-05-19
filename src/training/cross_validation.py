import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import yaml
import json
import torch
import random
import numpy as np
import argparse
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.models.classifier import CCClassifier
from src.utils.io import load_annotation_excel
from src.training.train_classifier import (
    get_patient_dirs, get_transforms, build_dataset,
    freeze_backbone, unfreeze_last_blocks, get_optimizer,
    train_one_epoch, evaluate,
)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_positive_frame_count(patient_dir, cc_frames_map):
    pid = patient_dir.name
    return len(cc_frames_map.get(pid, []))


def stratified_kfold_patients(patient_dirs, cc_frames_map, n_folds=5, seed=42):
    """
    Split patients into n_folds stratified by positive frame count and hospital site.
    Rare sites (< n_folds patients) are distributed round-robin.
    """
    random.seed(seed)

    patients = []
    for i, (p, d) in enumerate(patient_dirs):
        pid = p.name
        hospital = "-".join(pid.split("-")[:2])
        count = get_positive_frame_count(p, cc_frames_map)
        patients.append({"idx": i, "pid": pid, "hospital": hospital, "count": count})

    site_counts = Counter(p["hospital"] for p in patients)
    rare_sites = {site for site, cnt in site_counts.items() if cnt < n_folds}
    print(f"\nRare sites (< {n_folds} patients): {rare_sites}")

    rare_patients = [p for p in patients if p["hospital"] in rare_sites]
    common_patients = [p for p in patients if p["hospital"] not in rare_sites]

    random.shuffle(rare_patients)
    rare_folds = [[] for _ in range(n_folds)]
    for i, p in enumerate(rare_patients):
        rare_folds[i % n_folds].append(p["idx"])

    bins = {"zero": [], "low": [], "medium": [], "high": []}
    for p in common_patients:
        if p["count"] == 0:
            bins["zero"].append(p["idx"])
        elif p["count"] <= 5:
            bins["low"].append(p["idx"])
        elif p["count"] <= 15:
            bins["medium"].append(p["idx"])
        else:
            bins["high"].append(p["idx"])

    print("Common patient bins:")
    for bin_name, idxs in bins.items():
        print(f"  {bin_name}: {len(idxs)} patients")

    common_folds = [[] for _ in range(n_folds)]
    for bin_name, idxs in bins.items():
        random.shuffle(idxs)
        for i, idx in enumerate(idxs):
            common_folds[i % n_folds].append(idx)

    folds = [rare_folds[i] + common_folds[i] for i in range(n_folds)]
    print(f"\nFold sizes: {[len(f) for f in folds]}")

    splits = []
    for fold_idx in range(n_folds):
        val_indices = folds[fold_idx]
        train_indices = [idx for f in range(n_folds) if f != fold_idx for idx in folds[f]]
        splits.append((train_indices, val_indices))

    return splits


def train_fold(config, train_patient_dirs, val_patient_dirs, negative_frames_map,
               device, fold_idx, output_dir,
               seg_weight, use_aux_seg, seg_target_mode, seg_loss_type, mask_cfg):
    image_size = config["data"].get("image_size", 512)

    train_set = build_dataset(train_patient_dirs, negative_frames_map,
                              get_transforms(True, image_size), mask_cfg)
    val_set = build_dataset(val_patient_dirs, negative_frames_map,
                            get_transforms(False, image_size), mask_cfg)

    train_loader = DataLoader(train_set, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config["training"]["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=True)

    model = CCClassifier(
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        use_aux_seg=use_aux_seg,
    ).to(device)

    freeze_epochs = config["training"].get("freeze_epochs", 10)
    unfreeze_blocks = config["training"].get("unfreeze_blocks", 3)
    backbone_lr_scale = config["training"].get("backbone_lr_scale", 0.1)
    total_epochs = config["training"]["epochs"]
    patience = config["training"].get("early_stopping_patience", 11)

    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    best_val_auc = -1.0
    epochs_no_improve = 0
    phase = 1
    checkpoint_path = output_dir / f"fold_{fold_idx}_best.pth"

    for epoch in range(total_epochs):
        if epoch == freeze_epochs and phase == 1:
            print(f"  [Fold {fold_idx}] Epoch {epoch+1}: Phase 2 — unfreezing {unfreeze_blocks} blocks")
            unfreeze_last_blocks(model, num_blocks=unfreeze_blocks)
            optimizer = get_optimizer(model, config, backbone_lr_scale)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - freeze_epochs, eta_min=1e-7
            )
            epochs_no_improve = 0
            phase = 2

        tr_loss, tr_cls, tr_seg, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device,
            seg_weight, use_aux_seg, seg_target_mode, seg_loss_type,
        )
        vl_loss, vl_cls, vl_seg, vl_auc, _, _ = evaluate(
            model, val_loader, device,
            seg_weight, use_aux_seg, seg_target_mode, seg_loss_type,
        )

        if phase == 1:
            scheduler.step(vl_auc)
        else:
            scheduler.step()

        print(f"  [Fold {fold_idx}] Epoch {epoch+1}/{total_epochs} [Phase {phase}] "
              f"| Train loss {tr_loss:.4f} (cls {tr_cls:.4f} seg {tr_seg:.4f}) acc {tr_acc:.4f} "
              f"| Val loss {vl_loss:.4f} (cls {vl_cls:.4f} seg {vl_seg:.4f}) AUC {vl_auc:.4f}")

        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_auc": vl_auc,
                "val_loss": vl_loss,
            }, checkpoint_path)
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
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    excel_path = Path(config["data"]["annotation_excel"])
    patient_ids, cc_frames_map, negative_frames_map = load_annotation_excel(excel_path)

    sources = config["data"]["sources"]
    all_patient_dirs = get_patient_dirs(sources, patient_ids)
    print(f"Found {len(all_patient_dirs)} usable patients")

    loss_cfg = config.get("loss", {})
    use_aux_seg = loss_cfg.get("use_aux_seg", True)
    seg_weight = loss_cfg.get("seg_weight", 1.0)
    seg_target_mode = loss_cfg.get("seg_target_mode", "soft")
    seg_loss_type = loss_cfg.get("seg_loss_type", "dice_bce")
    mask_cfg = config.get("mask", {"inner_frac": 0.08, "outer_frac": 0.45})
    print(f"Aux seg head: {use_aux_seg} | seg_weight: {seg_weight} | "
          f"target: {seg_target_mode} | loss: {seg_loss_type}")

    cv_cfg = config.get("cv", {})
    n_folds = cv_cfg.get("n_folds", 5)
    pin_to_train = cv_cfg.get("pin_to_train", [])

    pinned = [(p, d) for p, d in all_patient_dirs if p.name in pin_to_train]
    foldable = [(p, d) for p, d in all_patient_dirs if p.name not in pin_to_train]

    print(f"Pinned to train always: {[p.name for p, _ in pinned]}")
    print(f"Patients available for CV folds: {len(foldable)}")

    splits = stratified_kfold_patients(foldable, cc_frames_map, n_folds=n_folds, seed=seed)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []

    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")

        train_patient_dirs = pinned + [foldable[i] for i in train_indices]
        val_patient_dirs = [foldable[i] for i in val_indices]

        print(f"  Train: {len(train_patient_dirs)} patients")
        print(f"  Val:   {[foldable[i][0].name for i in val_indices]}")

        checkpoint_path, val_loader, model = train_fold(
            config, train_patient_dirs, val_patient_dirs,
            negative_frames_map, device, fold_idx + 1, output_dir,
            seg_weight, use_aux_seg, seg_target_mode, seg_loss_type, mask_cfg,
        )

        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        _, _, _, auc, probs, labels = evaluate(
            model, val_loader, device,
            seg_weight, use_aux_seg, seg_target_mode, seg_loss_type,
        )
        preds = (probs > 0.5).astype(int)

        if len(np.unique(labels)) < 2:
            print(f"  [Fold {fold_idx+1}] Warning: only one class in val set, skipping AUC")
            auc = float("nan")

        metrics = {
            "fold": fold_idx + 1,
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

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    for metric in ["precision", "recall", "f1", "auc"]:
        values = [m[metric] for m in fold_metrics if not np.isnan(m[metric])]
        print(f"{metric.capitalize():12}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print(f"\nPer-fold breakdown:")
    print(f"{'Fold':>6} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8} {'Epoch':>7} {'Samples':>8}")
    for m in fold_metrics:
        print(f"{m['fold']:>6} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
              f"{m['auc']:>8.4f} {m['best_epoch']:>7} {m['n_val_samples']:>8}")

    results_path = output_dir / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump(fold_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()