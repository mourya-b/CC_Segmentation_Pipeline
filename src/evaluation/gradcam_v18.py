"""
Grad-CAM visualization for v18b — TP, TN, FP, FN categories.
Requires per-sample prediction CSV from evaluate_classifier.py.

Usage:
    python3 src/evaluation/gradcam_v18.py \
        --config /data/diag/mouryaBandaru/CC_Segmentation_Pipeline/configs/train_classifier_cluster.yaml \
        --checkpoint /data/diag/mouryaBandaru/experiments/classifier_v18b/fold_3_best.pth \
        --predictions-csv /data/diag/mouryaBandaru/experiments/classifier_v18b/predictions_fold_3.csv \
        --output-dir /data/diag/mouryaBandaru/gradcam_outputs/v18b_fold3 \
        --cv-splits /data/diag/mouryaBandaru/experiments/classifier_v18b/cv_splits.json \
        --fold 3 \
        --n-per-category 5
"""
import sys
from pathlib import Path
sys.path.insert(0, '/data/diag/mouryaBandaru/CC_Segmentation_Pipeline')

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import pydicom

from torch.utils.data import ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

from src.models.classifier import CCClassifier
from src.dataset.oct_cc_dataset import OCTFrameDataset
from src.utils.io import load_annotation_excel
from src.utils.masking import make_donut_mask


# ── Grad-CAM ──────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor):
        """image_tensor: (1, 3, H, W), normalized. Returns (cam HxW in [0,1], prob float)."""
        self.model.zero_grad()
        image_tensor = image_tensor.requires_grad_(True)
        cls_logit, _ = self.model(image_tensor)
        cls_logit.squeeze().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        prob = torch.sigmoid(cls_logit).item()
        return cam, prob


def get_target_layer(model):
    return model.encoder.conv_head


# ── Helpers ───────────────────────────────────────────────────────────────

def overlay_cam(image_uint8, cam, image_size, alpha=0.45):
    cam_resized = cv2.resize(cam, (image_size, image_size))
    heatmap = (plt.cm.jet(cam_resized)[..., :3] * 255).astype(np.uint8)
    return np.clip(alpha * heatmap + (1 - alpha) * image_uint8, 0, 255).astype(np.uint8)


def get_raw_image(pid, frame_idx, sources, mask_cfg, image_size):
    """Load raw frame, apply donut mask, resize to image_size."""
    dcm_path = None
    for src in sources:
        cand = Path(src["dicom_dir"]) / f"{pid}.dcm"
        if cand.exists():
            dcm_path = cand
            break
    if dcm_path is None:
        return None

    volume = pydicom.dcmread(str(dcm_path)).pixel_array
    raw = volume[frame_idx].astype(np.uint8)  # (H, W, 3)

    h, w = raw.shape[:2]
    donut = make_donut_mask(h, w, mask_cfg.get("inner_frac", 0.08),
                            mask_cfg.get("outer_frac", 0.45))
    raw = (raw * donut[..., None]).astype(np.uint8)
    raw = cv2.resize(raw, (image_size, image_size))
    return raw


def make_gt_overlay(raw_image, mask_tensor, image_size):
    """Red overlay of GT CC mask on raw image."""
    if mask_tensor is None or (hasattr(mask_tensor, 'sum') and mask_tensor.sum() == 0):
        return raw_image.copy()
    gt = mask_tensor.cpu().numpy() if isinstance(mask_tensor, torch.Tensor) else np.array(mask_tensor)
    gt = cv2.resize(gt, (image_size, image_size))
    red = np.zeros_like(raw_image)
    red[..., 0] = (gt > 0) * 255
    return np.clip(0.6 * raw_image + 0.4 * red, 0, 255).astype(np.uint8)


def save_figure(raw, gt_overlay, cam_overlay, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw);          axes[0].set_title("Image (masked)");         axes[0].axis("off")
    axes[1].imshow(gt_overlay);   axes[1].set_title("Ground Truth CC (red)");  axes[1].axis("off")
    axes[2].imshow(cam_overlay);  axes[2].set_title("Grad-CAM");               axes[2].axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close()


def categorize(df, threshold=0.5):
    df = df.copy()
    df["pred"] = (df["predicted_prob"] > threshold).astype(int)
    return {
        "TP": df[(df.true_label == 1) & (df.pred == 1)].sort_values("predicted_prob", ascending=False),
        "TN": df[(df.true_label == 0) & (df.pred == 0)].sort_values("predicted_prob", ascending=True),
        "FP": df[(df.true_label == 0) & (df.pred == 1)].sort_values("predicted_prob", ascending=False),
        "FN": df[(df.true_label == 1) & (df.pred == 0)].sort_values("predicted_prob", ascending=True),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cv-splits", default=None)
    parser.add_argument("--fold", default=None)
    parser.add_argument("--n-per-category", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_cfg = checkpoint.get("config", {})
    use_aux_seg = saved_cfg.get("loss", {}).get("use_aux_seg", True)
    mask_cfg = saved_cfg.get("mask", cfg.get("mask", {"inner_frac": 0.08, "outer_frac": 0.45}))

    model = CCClassifier(
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        use_aux_seg=use_aux_seg,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    gradcam = GradCAM(model, get_target_layer(model))

    # ── Val patient names ─────────────────────────────────────────────────
    val_patient_names = checkpoint.get("val_patients", None)
    if val_patient_names is None:
        if args.cv_splits and args.fold:
            with open(args.cv_splits) as f:
                cv_splits = json.load(f)
            val_patient_names = cv_splits.get(str(args.fold)) or cv_splits.get(int(args.fold))
        if val_patient_names is None:
            raise ValueError("No val_patients in checkpoint and no cv_splits provided.")
    print(f"Val patients: {val_patient_names}")

    # ── Build val dataset with metadata ───────────────────────────────────
    _, _, negative_frames_map = load_annotation_excel(cfg["data"]["annotation_excel"])
    sources = cfg["data"]["sources"]
    image_size = cfg["data"].get("image_size", 512)

    val_patient_dirs = []
    for pid in val_patient_names:
        hospital = "-".join(pid.split("-")[:2])
        for src in sources:
            base_dir = Path(src["base_dir"])
            dicom_dir = Path(src["dicom_dir"])
            patient_path = base_dir / hospital / pid
            dcm_path = dicom_dir / f"{pid}.dcm"
            if patient_path.exists() and dcm_path.exists():
                val_patient_dirs.append((patient_path, dicom_dir))
                break

    groups = {}
    for patient_dir, dicom_dir in val_patient_dirs:
        key = str(dicom_dir)
        groups.setdefault(key, {"dicom_dir": dicom_dir, "patient_dirs": []})
        groups[key]["patient_dirs"].append(patient_dir)

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    datasets = []
    for g in groups.values():
        ds = OCTFrameDataset(
            dicom_dir=g["dicom_dir"],
            patient_dirs=g["patient_dirs"],
            negative_frames_map=negative_frames_map,
            transform=transform,
            mask_inner_frac=mask_cfg.get("inner_frac", 0.08),
            mask_outer_frac=mask_cfg.get("outer_frac", 0.45),
            return_metadata=True,
        )
        datasets.append(ds)
    val_set = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # Build (patient_id, frame_idx) -> dataset index lookup
    print("Building lookup index...")
    lookup = {}
    for i in range(len(val_set)):
        _, mask, _, pid, fidx = val_set[i]
        lookup[(pid, int(fidx))] = (i, mask)
    print(f"Indexed {len(lookup)} samples")

    # ── Load predictions and categorize ───────────────────────────────────
    df = pd.read_csv(args.predictions_csv)
    cats = categorize(df, threshold=0.5)
    print("Category counts:", {k: len(v) for k, v in cats.items()})

    # ── Generate Grad-CAMs ────────────────────────────────────────────────
    for cat_name, cat_df in cats.items():
        cat_dir = output_dir / cat_name
        cat_dir.mkdir(exist_ok=True)
        print(f"\n=== {cat_name} ===")

        for rank, (_, row) in enumerate(cat_df.head(args.n_per_category).iterrows()):
            pid = row["patient_id"]
            fidx = int(row["frame_idx"])
            key = (pid, fidx)

            if key not in lookup:
                print(f"  #{rank}: {pid} fr{fidx} not in val set, skipping")
                continue

            sample_idx, mask_tensor = lookup[key]
            image_tensor, _, label, _, _ = val_set[sample_idx]
            image_in = image_tensor.unsqueeze(0).to(device)

            cam, prob = gradcam(image_in)

            raw = get_raw_image(pid, fidx, sources, mask_cfg, image_size)
            if raw is None:
                print(f"  #{rank}: could not load raw image for {pid}, skipping")
                continue

            gt_overlay = make_gt_overlay(raw, mask_tensor if int(label) == 1 else None, image_size)
            cam_overlay = overlay_cam(raw, cam, image_size)

            title = (f"{cat_name} #{rank} | {pid} fr{fidx} | "
                     f"label={int(label)} prob={prob:.3f}")
            out_path = cat_dir / f"{rank:02d}_{pid}_fr{fidx:04d}_prob{prob:.2f}.png"
            save_figure(raw, gt_overlay, cam_overlay, title, out_path)
            print(f"  #{rank}: {pid} fr{fidx} prob={prob:.3f} -> {out_path.name}")

    print(f"\nAll done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()