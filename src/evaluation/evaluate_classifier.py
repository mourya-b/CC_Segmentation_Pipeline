import sys
from pathlib import Path
sys.path.insert(0, '/data/diag/mouryaBandaru/CC_Segmentation_Pipeline')

import re
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve,
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

from src.models.classifier import CCClassifier
from src.dataset.oct_cc_dataset import OCTFrameDataset
from src.utils.io import load_annotation_excel


def get_val_patient_dirs(sources, val_patient_names):
    results = []
    for pid in val_patient_names:
        pid = str(pid).strip()
        if not pid or pid == "nan":
            continue
        hospital = "-".join(pid.split("-")[:2])
        for source in sources:
            base_dir = Path(source["base_dir"])
            dicom_dir = Path(source["dicom_dir"])
            patient_path = base_dir / hospital / pid
            dcm_path = dicom_dir / f"{pid}.dcm"
            if patient_path.exists() and dcm_path.exists():
                results.append((patient_path, dicom_dir))
                break
        else:
            print(f"  Warning: val patient {pid} not found in any source")
    return results


def build_val_dataset(val_patient_dirs_with_dicoms, negative_frames_map, transform,
                      mask_cfg, return_metadata=False):
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
            mask_inner_frac=mask_cfg.get("inner_frac", 0.08),
            mask_outer_frac=mask_cfg.get("outer_frac", 0.45),
            return_metadata=return_metadata,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def threshold_for_recall(probs, labels, target_recall=0.90):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    valid_idx = np.where(recall[:-1] >= target_recall)[0]
    if len(valid_idx) == 0:
        return None, float(recall.max()), float(precision[recall.argmax()])
    best_idx = valid_idx[np.argmax(precision[valid_idx])]
    return float(thresholds[best_idx]), float(recall[best_idx]), float(precision[best_idx])


def evaluate_checkpoint(checkpoint_path, cfg, device, output_csv_path,
                        fold_label="", cv_splits=None, fold_num=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Priority 1: val_patients saved in checkpoint
    val_patient_names = checkpoint.get("val_patients", None)

    # Priority 2: external cv_splits.json
    if val_patient_names is None and cv_splits is not None and fold_num is not None:
        val_patient_names = cv_splits.get(str(fold_num)) or cv_splits.get(fold_num)
        if val_patient_names:
            print(f"  Using val patients from cv_splits.json for fold {fold_num}")

    if val_patient_names is None:
        print(f"  ERROR: no val_patients for {checkpoint_path.name}. "
              f"Run extract_cv_splits.py to generate cv_splits.json.")
        return None

    saved_cfg = checkpoint.get("config", {})
    mask_cfg = saved_cfg.get("mask", cfg.get("mask", {"inner_frac": 0.08, "outer_frac": 0.45}))
    use_aux_seg = saved_cfg.get("loss", {}).get("use_aux_seg", True)

    _, _, negative_frames_map = load_annotation_excel(cfg["data"]["annotation_excel"])
    sources = cfg["data"]["sources"]
    val_patient_dirs = get_val_patient_dirs(sources, val_patient_names)

    if not val_patient_dirs:
        print(f"  ERROR: no val patient dirs found for {fold_label}")
        return None

    image_size = cfg["data"].get("image_size", 512)
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_set = build_val_dataset(val_patient_dirs, negative_frames_map, val_transform,
                                mask_cfg, return_metadata=True)
    val_loader = DataLoader(val_set, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=0)

    model = CCClassifier(
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        use_aux_seg=use_aux_seg,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    records = []
    with torch.no_grad():
        for images, masks, labels, patient_ids_batch, frame_idxs in val_loader:
            images = images.to(device)
            cls_logits, _ = model(images)
            probs = torch.sigmoid(cls_logits).cpu().numpy()
            labels_np = labels.numpy().astype(int)
            frame_idxs_np = frame_idxs.numpy() if isinstance(frame_idxs, torch.Tensor) \
                            else np.array(frame_idxs)

            for pid, fidx, prob, lbl in zip(patient_ids_batch, frame_idxs_np, probs, labels_np):
                records.append({
                    "patient_id": pid,
                    "frame_idx": int(fidx),
                    "true_label": int(lbl),
                    "predicted_prob": float(prob),
                })

    df = pd.DataFrame(records)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    all_probs = df["predicted_prob"].values
    all_labels = df["true_label"].values
    all_preds = (all_probs > 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"\n{'='*60}")
    print(f"{fold_label}  |  epoch {checkpoint.get('epoch','?')}  |  saved AUC {checkpoint.get('val_auc', '?'):.4f}")
    print(f"  Val patients: {val_patient_names}")
    print(f"  Samples: {len(df)}  (pos {int(all_labels.sum())}, neg {int((1-all_labels).sum())})")
    print(f"  AUC:        {auc:.4f}")
    print(f"  Prec@0.5:   {prec:.4f}")
    print(f"  Recall@0.5: {rec:.4f}")
    print(f"  F1@0.5:     {f1:.4f}")
    print(f"  Threshold tuning:")
    for target in [0.85, 0.90, 0.95]:
        thr, r, p = threshold_for_recall(all_probs, all_labels, target)
        if thr is not None:
            print(f"    Recall>={target}: threshold={thr:.3f}  recall={r:.3f}  precision={p:.3f}")
        else:
            print(f"    Recall>={target}: not achievable (max recall={r:.3f})")

    return {
        "fold_label": fold_label,
        "auc": auc, "precision": prec, "recall": rec, "f1": f1,
        "n_samples": len(df), "n_pos": int(all_labels.sum()),
        "val_patients": val_patient_names,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_classifier_cluster.yaml")
    parser.add_argument("--exp-dir", required=True,
                        help="Experiment dir containing fold_N_best.pth files or best_classifier.pth")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save prediction CSVs. Defaults to --exp-dir.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir

    # Load cv_splits.json if present
    cv_splits = None
    cv_splits_path = exp_dir / "cv_splits.json"
    if cv_splits_path.exists():
        with open(cv_splits_path) as f:
            cv_splits = json.load(f)
        print(f"Loaded CV splits from {cv_splits_path}")
    else:
        print(f"No cv_splits.json found in {exp_dir}. "
              f"Checkpoints must contain val_patients key, or run extract_cv_splits.py first.")

    # Find fold checkpoints
    fold_ckpts = sorted(exp_dir.glob("fold_*_best.pth"))
    if not fold_ckpts:
        single = exp_dir / "best_classifier.pth"
        fold_ckpts = [single] if single.exists() else []

    if not fold_ckpts:
        print(f"No checkpoints found in {exp_dir}")
        return

    summaries = []
    for ckpt_path in fold_ckpts:
        fold_num_match = re.search(r"fold_(\d+)", ckpt_path.stem)
        fold_num = int(fold_num_match.group(1)) if fold_num_match else None
        fold_label = ckpt_path.stem.replace("_best", "")
        csv_path = output_dir / f"predictions_{fold_label}.csv"
        s = evaluate_checkpoint(ckpt_path, cfg, device, csv_path,
                                fold_label=fold_label, cv_splits=cv_splits, fold_num=fold_num)
        if s:
            summaries.append(s)

    if len(summaries) > 1:
        print(f"\n{'='*60}")
        print("CV SUMMARY")
        print(f"{'='*60}")
        aucs = [s["auc"] for s in summaries]
        recs = [s["recall"] for s in summaries]
        precs = [s["precision"] for s in summaries]
        f1s = [s["f1"] for s in summaries]
        print(f"AUC:        {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"Recall@0.5: {np.mean(recs):.4f} ± {np.std(recs):.4f}")
        print(f"Prec@0.5:   {np.mean(precs):.4f} ± {np.std(precs):.4f}")
        print(f"F1@0.5:     {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"\nPer-fold:")
        for s in summaries:
            print(f"  {s['fold_label']:20s} AUC {s['auc']:.4f}  "
                  f"n={s['n_samples']}  pos={s['n_pos']}")


if __name__ == "__main__":
    main()