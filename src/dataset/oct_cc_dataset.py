import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from src.utils.io import load_segmentation, extract_cc_mask
from src.utils.masking import make_donut_mask
import pydicom


class OCTFrameDataset(Dataset):
    def __init__(self, dicom_dir, patient_dirs, negative_frames_map=None, transform=None,
                 mask_inner_frac=0.08, mask_outer_frac=0.45):
        self.transform = transform
        self.samples = []
        self.volume_cache = {}
        self.seg_cache = {}
        self.mask_inner_frac = mask_inner_frac
        self.mask_outer_frac = mask_outer_frac
        self._cached_donut = None
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

    def _load_cc_mask_volume(self, nii_path):
        key = str(nii_path)
        if key not in self.seg_cache:
            seg = load_segmentation(nii_path)
            cc_mask = extract_cc_mask(seg)  # (N, H, W) bool
            self.seg_cache[key] = cc_mask.astype(np.float32)
        return self.seg_cache[key]

    def _get_donut(self, h, w):
        if self._cached_donut is None or self._cached_donut.shape[:2] != (h, w):
            mask_2d = make_donut_mask(h, w, self.mask_inner_frac, self.mask_outer_frac)
            self._cached_donut = mask_2d  # (H, W) float32
        return self._cached_donut

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dcm_path, nii_path, frame_idx, label = self.samples[idx]

        volume = self._load_volume(dcm_path)
        image = volume[frame_idx]                          # (H, W, 3) uint8

        cc_vol = self._load_cc_mask_volume(nii_path)
        mask = cc_vol[frame_idx]                           # (H, W) float32

        # Apply donut to both
        donut = self._get_donut(image.shape[0], image.shape[1])
        image = (image * donut[..., None]).astype(image.dtype)
        mask = mask * donut

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32)

        # Ensure mask is a float tensor (albumentations may keep it numpy if no ToTensorV2 path)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        else:
            mask = mask.float()

        return image, mask, torch.tensor(label, dtype=torch.float32)