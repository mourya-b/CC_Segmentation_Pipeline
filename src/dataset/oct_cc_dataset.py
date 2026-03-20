import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from src.utils.io import load_segmentation, extract_cc_mask
import pydicom


class OCTFrameDataset(Dataset):
    def __init__(self, dicom_dir, patient_dirs, negative_frames_map=None, transform=None):
        """
        dicom_dir: Path to directory containing {PatientID}.dcm files
        patient_dirs: list of Path objects pointing to each patient folder (for nii files)
        negative_frames_map: dict mapping patient_id to list of negative frame indices (0-indexed)
        transform: albumentations transform pipeline
        """
        self.transform = transform
        self.samples = []
        self.volume_cache = {}
        negative_frames_map = negative_frames_map or {}
        dicom_dir = Path(dicom_dir)

        for patient_dir in patient_dirs:
            patient_dir = Path(patient_dir)
            nii_files = list(patient_dir.glob("*_CC.nii.gz"))
            patient_id = patient_dir.name
            dcm_path = dicom_dir / f"{patient_id}.dcm"

            if not nii_files or not dcm_path.exists():
                print(f"Skipping {patient_id}: missing dcm or nii file")
                continue

            nii_path = nii_files[0]
            seg = load_segmentation(nii_path)
            cc_mask = extract_cc_mask(seg)

            cc_frames = set(np.where(cc_mask.any(axis=(1, 2)))[0].tolist())
            for frame_idx in cc_frames:
                self.samples.append((dcm_path, nii_path, frame_idx, 1))

            neg_frames = negative_frames_map.get(patient_id, [])
            for frame_idx in neg_frames:
                if frame_idx not in cc_frames:
                    self.samples.append((dcm_path, nii_path, frame_idx, 0))

        print(f"Total samples: {len(self.samples)}")
        print(f"Positive (CC): {sum(s[3] == 1 for s in self.samples)}")
        print(f"Negative: {sum(s[3] == 0 for s in self.samples)}")

    def _load_volume(self, dcm_path):
        key = str(dcm_path)
        if key not in self.volume_cache:
            dcm = pydicom.dcmread(str(dcm_path))
            self.volume_cache[key] = dcm.pixel_array  # (N, H, W, 3)
        return self.volume_cache[key]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dcm_path, nii_path, frame_idx, label = self.samples[idx]

        volume = self._load_volume(dcm_path)
        image = volume[frame_idx]  # (H, W, 3) already RGB

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        return image, torch.tensor(label, dtype=torch.long)