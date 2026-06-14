"""
Convert annotated DICOM frames + .nii.gz masks into nnU-Net v2 format.

Output structure:
  output_dir/
    Dataset001_CC/
      dataset.json
      imagesTr/
        {pid}_fr{idx:04d}_0000.png   # R channel
        {pid}_fr{idx:04d}_0001.png   # G channel
        {pid}_fr{idx:04d}_0002.png   # B channel
      labelsTr/
        {pid}_fr{idx:04d}.png        # binary mask
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

from src.utils.io import load_annotation_excel, load_segmentation, extract_cc_mask


def find_patient_paths(pid, sources):
    hospital = "-".join(pid.split("-")[:2])
    for src in sources:
        base_dir = Path(src["base_dir"])
        dicom_dir = Path(src["dicom_dir"])
        seg_dir = base_dir / hospital / pid
        dcm_path = dicom_dir / f"{pid}.dcm"
        if dcm_path.exists() and seg_dir.exists():
            return dcm_path, seg_dir
    return None, None


def save_frame_and_label(frame_rgb, cc_mask, out_img_dir, out_lbl_dir, pid, frame_idx):
    """
    Save an RGB frame as 3 separate channel PNGs (nnU-Net v2 convention)
    and the binary label mask as one PNG.
    """
    case_id = f"{pid}_fr{frame_idx:04d}"
    # Save each channel as a separate PNG
    for c in range(3):
        Image.fromarray(frame_rgb[:, :, c].astype(np.uint8)).save(
            out_img_dir / f"{case_id}_{c:04d}.png"
        )
    # Save label as binary 0/1 PNG
    label = (cc_mask > 0).astype(np.uint8)
    Image.fromarray(label).save(out_lbl_dir / f"{case_id}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="train_classifier config yaml (for data sources)")
    parser.add_argument("--output-dir", required=True,
                        help="nnU-Net raw dataset root (e.g., /data/diag/.../nnUNet_raw)")
    parser.add_argument("--dataset-id", type=int, default=1)
    parser.add_argument("--dataset-name", default="CC")
    parser.add_argument("--include-negatives", action="store_true",
                        help="Also include CC-negative frames with empty masks")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sources = cfg["data"]["sources"]

    patient_ids, _, negative_frames_map = load_annotation_excel(cfg["data"]["annotation_excel"])

    dataset_dir = Path(args.output_dir) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    img_dir = dataset_dir / "imagesTr"
    lbl_dir = dataset_dir / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_pos, n_neg, missing = 0, 0, []
    case_ids = []  # for the dataset.json

    for pid in tqdm(patient_ids, desc="Patients"):
        pid = str(pid).strip()
        if not pid or pid == "nan":
            continue

        dcm_path, seg_dir = find_patient_paths(pid, sources)
        if dcm_path is None:
            missing.append(pid)
            continue

        nii_files = list(seg_dir.glob("*_CC.nii.gz"))
        if not nii_files:
            missing.append(pid)
            continue

        # Load CC mask volume
        seg = load_segmentation(nii_files[0])
        cc_vol = extract_cc_mask(seg)  # (N, H, W) bool

        # Load DICOM volume
        try:
            volume = pydicom.dcmread(str(dcm_path)).pixel_array  # (N, H, W, 3)
        except Exception as e:
            print(f"Failed reading {pid}: {e}")
            missing.append(pid)
            continue

        # Verify shapes match
        if volume.shape[0] != cc_vol.shape[0]:
            print(f"Frame count mismatch for {pid}: dicom={volume.shape[0]} mask={cc_vol.shape[0]}")
            missing.append(pid)
            continue

        # Positive frames
        cc_frame_idxs = np.where(cc_vol.any(axis=(1, 2)))[0]
        for fidx in cc_frame_idxs:
            save_frame_and_label(
                volume[fidx], cc_vol[fidx],
                img_dir, lbl_dir, pid, int(fidx),
            )
            case_ids.append(f"{pid}_fr{int(fidx):04d}")
            n_pos += 1

        # Negative frames (only if requested)
        if args.include_negatives:
            cc_set = set(cc_frame_idxs.tolist())
            for fidx in negative_frames_map.get(pid, []):
                if fidx in cc_set or fidx >= len(volume):
                    continue
                empty_mask = np.zeros(volume.shape[1:3], dtype=bool)
                save_frame_and_label(
                    volume[fidx], empty_mask,
                    img_dir, lbl_dir, pid, int(fidx),
                )
                case_ids.append(f"{pid}_fr{int(fidx):04d}")
                n_neg += 1

        # Free volume
        del volume, cc_vol

    # Build dataset.json
    dataset_json = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": {
            "background": 0,
            "CC": 1
        },
        "numTraining": len(case_ids),
        "file_ending": ".png",
        "name": args.dataset_name,
        "description": "Cholesterol crystal segmentation from intracoronary OCT frames"
    }

    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDone.")
    print(f"  Positive frames: {n_pos}")
    print(f"  Negative frames: {n_neg}")
    print(f"  Total cases: {len(case_ids)}")
    print(f"  Missing/skipped patients: {len(missing)}")
    if missing:
        print(f"  Missing list: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    print(f"  Output: {dataset_dir}")


if __name__ == "__main__":
    main()