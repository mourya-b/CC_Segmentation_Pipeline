import pydicom
import nibabel as nib
import numpy as np
import zarr

def load_dicom_volume(dcm_path):
    """Load a multi-frame DICOM and return array of shape (N, H, W)."""
    dcm = pydicom.dcmread(dcm_path)
    pixel_array = dcm.pixel_array  # shape: (N, H, W)
    return pixel_array

def load_zarr_volume(zarr_path):
    """Load a Zarr volume and return array of shape (N, H, W, 3)."""
    z = zarr.open(str(zarr_path), mode='r')
    volume = z['data']  # (N, 3, H, W)
    return volume  # keep as zarr array for lazy loading

def load_segmentation(nii_path):
    """Load a .nii.gz segmentation and return array of shape (N, H, W), aligned to DICOM orientation."""
    nii = nib.load(nii_path)
    seg = nii.get_fdata().astype(np.uint8)
    seg = seg.transpose(2, 0, 1)
    seg = np.rot90(seg, k=3, axes=(1, 2))
    seg = np.flip(seg, axis=2)
    return np.ascontiguousarray(seg)


def extract_cc_mask(seg_volume, cc_label=15):
    """Extract binary CC mask from segmentation volume. Returns (N, H, W)."""
    return (seg_volume == cc_label).astype(np.uint8)

import pandas as pd


def parse_frame_list(cell_value):
    """Parse a comma-separated string of frame numbers into a 0-indexed list."""
    if not cell_value or str(cell_value).strip() == "":
        return []
    parts = str(cell_value).split(",")
    result = []
    for p in parts:
        p = p.strip()
        if p.isdigit():
            result.append(int(p) - 1)  # convert to 0-indexed
    return result


def parse_macrophage_ranges(cell_value):
    """Parse macrophage range strings like '363-388, 106-110' into 0-indexed frame lists."""
    if not cell_value or str(cell_value).strip() == "":
        return []
    result = []
    parts = str(cell_value).split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) == 2 and bounds[0].strip().isdigit() and bounds[1].strip().isdigit():
                start, end = int(bounds[0].strip()), int(bounds[1].strip())
                result.extend(range(start - 1, end))  # 0-indexed
    return result


def load_annotation_excel(excel_path):
    """
    Read annotation excel and return:
    - patient_ids: list of patient Full_Filenames
    - cc_frames_map: dict {patient_id: [0-indexed cc frame indices]}
    - negative_frames_map: dict {patient_id: [0-indexed negative + hard negative frame indices]}
    """
    df = pd.read_excel(excel_path, sheet_name="CC Annotations")
    df = df[df["Full_Filename"].notna()]

    cc_frames_map = {}
    negative_frames_map = {}

    for _, row in df.iterrows():
        pid = row["Full_Filename"]
        cc_frames_map[pid] = parse_frame_list(row["Frames_CC"])
        neg_frames = parse_frame_list(row["Frames_Negative"])
        macro_frames = parse_macrophage_ranges(row["Macrophages"])
        # combine negatives and hard negatives, remove duplicates
        negative_frames_map[pid] = list(set(neg_frames + macro_frames))

    return list(df["Full_Filename"]), cc_frames_map, negative_frames_map