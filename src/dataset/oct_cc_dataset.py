import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from src.utils.io import load_segmentation, extract_cc_mask
import zarr


class OCTFrameDataset(Dataset):
    """
    Frame-level dataset for CC classification.
    Each item is a single OCT frame with a binary label (1=CC present, 0=negative).
    """

    def __init__(self, zarr_dir, patient_dirs, negative_frames_map=None, transform=None):
        """
        zarr_dir: Path to the directory containing all {PatientID}.zarr folders
        patient_dirs: list of Path objects pointing to each patient folder (for nii files)
        negative_frames_map: dict mapping patient_id (str) to list of negative frame indices (0-indexed)
        transform: albumentations transform pipeline
        """
        self.transform = transform
        self.samples = []
        negative_frames_map = negative_frames_map or {}
        zarr_dir = Path(zarr_dir)

        for patient_dir in patient_dirs:
            patient_dir = Path(patient_dir)
            nii_files = list(patient_dir.glob("*_CC.nii.gz"))
            patient_id = patient_dir.name
            zarr_path = zarr_dir / f"{patient_id}.zarr"

            if not nii_files or not zarr_path.exists():
                print(f"Skipping {patient_id}: missing zarr or nii file")
                continue

            nii_path = nii_files[0]

            seg = load_segmentation(nii_path)
            cc_mask = extract_cc_mask(seg)

            cc_frames = set(np.where(cc_mask.any(axis=(1, 2)))[0].tolist())
            for frame_idx in cc_frames:
                self.samples.append((zarr_path, nii_path, frame_idx, 1))

            neg_frames = negative_frames_map.get(patient_id, [])
            for frame_idx in neg_frames:
                if frame_idx not in cc_frames:
                    self.samples.append((zarr_path, nii_path, frame_idx, 0))

        print(f"Total samples: {len(self.samples)}")
        print(f"Positive (CC): {sum(s[3] == 1 for s in self.samples)}")
        print(f"Negative: {sum(s[3] == 0 for s in self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        zarr_path, nii_path, frame_idx, label = self.samples[idx]

        z = zarr.open(str(zarr_path), mode='r')
        image = z['data'][frame_idx]  # (3, H, W)
        image = np.transpose(image, (1, 2, 0))  # -> (H, W, 3)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        return image, torch.tensor(label, dtype=torch.long)