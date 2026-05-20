"""
Grad-CAM diagnostic with ground truth overlay and negative comparison.

Produces per-sample PNGs with 3 panels for positives:
    [raw frame | GT CC mask overlay | Grad-CAM overlay]
And 2 panels for negatives:
    [raw frame | Grad-CAM overlay]

Usage:
    python src/evaluation/gradcam_diagnostic.py \
        --checkpoint /data/diag/mouryaBandaru/experiments/classifier_v18/best_classifier.pth \
        --config /data/diag/mouryaBandaru/CC_Segmentation_Pipeline/configs/train_classifier_cluster.yaml \
        --patients NLD-AZN-0009 NLD-AZN-0012 ABW-ARU-0003 \
        --label v18_fold3 \
        --n_pos 5 --n_neg 3 \
        --output_dir /data/diag/mouryaBandaru/gradcam_outputs
"""
import sys
from pathlib import Path
sys.path.insert(0, '/data/diag/mouryaBandaru/CC_Segmentation_Pipeline')

import argparse
import yaml
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.classifier import CCClassifier
from src.utils.io import load_annotation_excel
from src.training.train_classifier import get_patient_dirs, build_dataset


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, input, output):
        self.activations = output.detach()

    def _bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        cls_logit, _ = self.model(input_tensor)
        self.model.zero_grad()
        cls_logit.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        prob = torch.sigmoid(cls_logit).item()
        return cam, prob


def get_target_layer(model):
    return list(model.encoder.blocks.children())[-1]


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = tensor.detach().squeeze().cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean) * 255
    return np.clip(img, 0, 255).astype(np.uint8)


def save_positive(img_np, gt_mask_np, cam, prob, label, save_path):
    """3-panel: raw frame | GT mask overlay | Grad-CAM overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{label} | prob={prob:.3f}", fontsize=13)

    axes[0].imshow(img_np)
    axes[0].set_title("Frame")
    axes[0].axis("off")

    axes[1].imshow(img_np)
    if gt_mask_np.any():
        axes[1].imshow(gt_mask_np, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Ground Truth CC")
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Grad-CAM")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_negative(img_np, cam, prob, label, save_path):
    """2-panel: raw frame | Grad-CAM overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{label} | prob={prob:.3f} [NEGATIVE]", fontsize=13)

    axes[0].imshow(img_np)
    axes[0].set_title("Frame")
    axes[0].axis("off")

    axes[1].imshow(img_np)
    axes[1].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",
                        default="/data/diag/mouryaBandaru/CC_Segmentation_Pipeline/configs/train_classifier_cluster.yaml")
    parser.add_argument("--patients", nargs="+", required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--n_pos", type=int, default=5,
                        help="Max positive frames to save")
    parser.add_argument("--n_neg", type=int, default=3,
                        help="Max negative frames to save")
    parser.add_argument("--output_dir",
                        default="/data/diag/mouryaBandaru/gradcam_outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"Checkpoint epoch: {ckpt.get('epoch','?')} | Val AUC: {ckpt.get('val_auc','?')}")

    model = CCClassifier(
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        use_aux_seg=cfg.get("loss", {}).get("use_aux_seg", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    gradcam = GradCAM(model, get_target_layer(model))

    excel_path = cfg["data"]["annotation_excel"]
    patient_ids, cc_frames_map, negative_frames_map = load_annotation_excel(excel_path)

    sources = cfg["data"]["sources"]
    all_dirs = get_patient_dirs(sources, patient_ids)

    target_set = set(args.patients)
    selected_dirs = [(p, d) for p, d in all_dirs if p.name in target_set]
    missing = target_set - {p.name for p, _ in selected_dirs}
    if missing:
        print(f"Warning: not found on disk: {missing}")
    print(f"Patients: {[p.name for p, _ in selected_dirs]}")

    image_size = cfg["data"].get("image_size", 512)
    mask_cfg = cfg.get("mask", {"inner_frac": 0.08, "outer_frac": 0.45})

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    dataset = build_dataset(selected_dirs, negative_frames_map, transform, mask_cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    out_pos = Path(args.output_dir) / args.label / "positives"
    out_neg = Path(args.output_dir) / args.label / "negatives"
    out_pos.mkdir(parents=True, exist_ok=True)
    out_neg.mkdir(parents=True, exist_ok=True)

    pos_count = neg_count = 0
    print(f"\nGenerating: {args.n_pos} positives, {args.n_neg} negatives...")

    for images, masks, labels in loader:
        if pos_count >= args.n_pos and neg_count >= args.n_neg:
            break

        is_positive = labels.item() == 1

        if is_positive and pos_count >= args.n_pos:
            continue
        if not is_positive and neg_count >= args.n_neg:
            continue

        images_dev = images.to(device)
        cam, prob = gradcam.generate(images_dev)
        img_np = denormalize(images)

        if is_positive:
            gt_mask = masks.squeeze().cpu().numpy()  # (H, W) float32
            # Resize gt_mask to match image size if needed
            if gt_mask.shape != img_np.shape[:2]:
                gt_mask = cv2.resize(gt_mask, (img_np.shape[1], img_np.shape[0]))

            fname = out_pos / f"pos_{pos_count:03d}_prob{prob:.3f}.png"
            save_positive(img_np, gt_mask, cam, prob,
                          label=f"{args.label} | pos {pos_count}", save_path=fname)
            pred_str = "TP" if prob > 0.5 else "FN"
            print(f"  {pred_str} pos_{pos_count:03d}: prob={prob:.3f} | GT mask pixels: {int(gt_mask.sum())}")
            pos_count += 1
        else:
            fname = out_neg / f"neg_{neg_count:03d}_prob{neg_count:.3f}.png"
            fname = out_neg / f"neg_{neg_count:03d}_prob{prob:.3f}.png"
            save_negative(img_np, cam, prob,
                          label=f"{args.label} | neg {neg_count}", save_path=fname)
            pred_str = "TN" if prob <= 0.5 else "FP"
            print(f"  {pred_str} neg_{neg_count:03d}: prob={prob:.3f}")
            neg_count += 1

    print(f"\nDone.")
    print(f"  Positives -> {out_pos}")
    print(f"  Negatives -> {out_neg}")


if __name__ == "__main__":
    main()