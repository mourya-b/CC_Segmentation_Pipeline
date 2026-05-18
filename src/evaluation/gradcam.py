# src/evaluation/gradcam.py
import sys
from pathlib import Path
sys.path.insert(0, '/data/diag/mouryaBandaru/CC_Segmentation_Pipeline')

import torch
import numpy as np
import cv2
import os
import argparse
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.classifier import CCClassifier
from src.dataset.oct_cc_dataset import OCTFrameDataset
from src.utils.io import load_annotation_excel


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=1):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Pool gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, torch.softmax(output, dim=1)[0, 1].item()


def overlay_cam(image_np, cam, alpha=0.4):
    """Overlay CAM heatmap on original image."""
    # image_np: HxWx3 uint8
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * image_np).astype(np.uint8)
    return overlay


def get_target_layer(model):
    """Get the last convolutional layer for Grad-CAM."""
    backbone = model.backbone_name
    if "efficientnet" in backbone:
        # Last block of EfficientNet
        blocks = list(model.model.blocks.children())
        return blocks[-1]
    elif "resnet" in backbone:
        return model.model.layer4[-1]
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/train_classifier_cluster.yaml")
    parser.add_argument("--output_dir", default="/data/diag/mouryaBandaru/gradcam_outputs")
    parser.add_argument("--n_samples", type=int, default=20)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    val_patients = checkpoint.get("val_patients", [])
    print(f"Val patients: {val_patients}")

    # Load model
    model = CCClassifier(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Get target layer
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    # Load data
    excel_path = cfg["data"]["annotation_excel"]
    patient_ids, cc_frames_map, negative_frames_map = load_annotation_excel(excel_path)

    # Build val dataset
    from src.training.train_classifier import get_patient_dirs, build_dataset
    sources = cfg["data"]["sources"]
    all_dirs = get_patient_dirs(sources, patient_ids)
    val_dirs = [(p, d) for p, d in all_dirs if p.name in val_patients]

    image_size = cfg["data"].get("image_size", 512)
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_set = build_dataset(val_dirs, negative_frames_map, transform)
    loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    # Output dirs
    out_correct = Path(args.output_dir) / "correct_positives"
    out_wrong = Path(args.output_dir) / "wrong_positives"
    out_correct.mkdir(parents=True, exist_ok=True)
    out_wrong.mkdir(parents=True, exist_ok=True)

    correct_count = 0
    wrong_count = 0

    print(f"\nGenerating Grad-CAM for {args.n_samples} correct and {args.n_samples} wrong positives...")

    for i, (image, label) in enumerate(loader):
        if correct_count >= args.n_samples and wrong_count >= args.n_samples:
            break

        label_val = label.item()
        if label_val != 1:  # only process positives
            continue

        image = image.to(device)
        cam, prob = gradcam.generate(image, class_idx=1)
        pred = 1 if prob > 0.5 else 0

        # Denormalise image for saving
        img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (img_np * std + mean) * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        overlay = overlay_cam(img_np, cam)

        if pred == 1 and correct_count < args.n_samples:
            # Correctly classified positive
            save_path = out_correct / f"sample_{correct_count:03d}_prob{prob:.2f}.png"
            combined = np.concatenate([img_np, overlay], axis=1)
            cv2.imwrite(str(save_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            correct_count += 1
        elif pred == 0 and wrong_count < args.n_samples:
            # Wrongly classified positive (false negative)
            save_path = out_wrong / f"sample_{wrong_count:03d}_prob{prob:.2f}.png"
            combined = np.concatenate([img_np, overlay], axis=1)
            cv2.imwrite(str(save_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            wrong_count += 1

    print(f"\nSaved {correct_count} correct positives to {out_correct}")
    print(f"Saved {wrong_count} wrong positives (false negatives) to {out_wrong}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()