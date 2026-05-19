"""
Targeted Grad-CAM for specific patient lists.
Usage:
    python src/evaluation/gradcam_targeted.py \
        --checkpoint /data/diag/mouryaBandaru/experiments/classifier_v17/best_classifier.pth \
        --patients NLD-RADB-0002 NLD-TERG-0003 NLD-ISALA-0049 \
        --label fold3 \
        --n_samples 5 \
        --output_dir /data/diag/mouryaBandaru/gradcam_outputs/v17
"""
import sys
from pathlib import Path
sys.path.insert(0, '/data/diag/mouryaBandaru/CC_Segmentation_Pipeline')

import argparse
import yaml
import numpy as np
import cv2
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
        """Returns (cam, prob) where cam is HxW in [0,1] and prob is sigmoid score."""
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
    blocks = list(model.encoder.blocks.children())
    return blocks[-1]


def overlay_cam(image_np, cam, alpha=0.4):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.clip(alpha * heatmap + (1 - alpha) * image_np, 0, 255).astype(np.uint8)


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean) * 255
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/train_classifier_cluster.yaml")
    parser.add_argument("--patients", nargs="+", required=True,
                        help="Patient IDs to visualize")
    parser.add_argument("--label", default="fold",
                        help="Tag for output subdirectory (e.g. fold3, fold5)")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Max positive frames to save per patient")
    parser.add_argument("--output_dir",
                        default="/data/diag/mouryaBandaru/gradcam_outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')} | Val AUC: {ckpt.get('val_auc', '?')}")

    model = CCClassifier(
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        use_aux_seg=cfg.get("loss", {}).get("use_aux_seg", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    # Load only the requested patients
    excel_path = cfg["data"]["annotation_excel"]
    patient_ids, cc_frames_map, negative_frames_map = load_annotation_excel(excel_path)

    sources = cfg["data"]["sources"]
    all_dirs = get_patient_dirs(sources, patient_ids)

    target_set = set(args.patients)
    selected_dirs = [(p, d) for p, d in all_dirs if p.name in target_set]
    found = {p.name for p, _ in selected_dirs}
    missing = target_set - found
    if missing:
        print(f"Warning: these patients not found on disk: {missing}")
    print(f"Running Grad-CAM on {len(selected_dirs)} patients: {[p.name for p, _ in selected_dirs]}")

    image_size = cfg["data"].get("image_size", 512)
    mask_cfg = cfg.get("mask", {"inner_frac": 0.08, "outer_frac": 0.45})

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    dataset = build_dataset(selected_dirs, negative_frames_map, transform, mask_cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    out_pos = Path(args.output_dir) / args.label / "true_positives"
    out_fn  = Path(args.output_dir) / args.label / "false_negatives"
    out_pos.mkdir(parents=True, exist_ok=True)
    out_fn.mkdir(parents=True, exist_ok=True)

    tp_count = fn_count = 0
    print(f"\nGenerating Grad-CAMs (max {args.n_samples} positive frames total)...")

    for images, masks, labels in loader:
        if tp_count + fn_count >= args.n_samples * 2:
            break
        if labels.item() != 1:
            continue

        images = images.to(device)
        cam, prob = gradcam.generate(images)
        pred = 1 if prob > 0.5 else 0

        img_np = denormalize(images)
        overlay = overlay_cam(img_np, cam)
        combined = np.concatenate([img_np, overlay], axis=1)

        if pred == 1 and tp_count < args.n_samples:
            fname = out_pos / f"tp_{tp_count:03d}_prob{prob:.2f}.png"
            cv2.imwrite(str(fname), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            tp_count += 1
            print(f"  TP saved: prob={prob:.3f}")
        elif pred == 0 and fn_count < args.n_samples:
            fname = out_fn / f"fn_{fn_count:03d}_prob{prob:.2f}.png"
            cv2.imwrite(str(fname), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            fn_count += 1
            print(f"  FN saved: prob={prob:.3f}")

    print(f"\nDone. TPs: {tp_count} -> {out_pos}")
    print(f"      FNs: {fn_count} -> {out_fn}")


if __name__ == "__main__":
    main()