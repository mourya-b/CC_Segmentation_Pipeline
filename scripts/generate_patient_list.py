from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.io import load_annotation_excel

excel_path = "/data/diag/mouryaBandaru/data/CC_Annotations_v3.xlsx"
patient_ids, _, _ = load_annotation_excel(excel_path)

out_path = Path(__file__).parent / "patient_list.txt"
written = 0
with open(out_path, "w") as f:
    for pid in patient_ids:
        pid = str(pid).strip()
        if pid and pid != "nan":
            f.write(pid + "\n")
            written += 1

print(f"Wrote {written} patient IDs to {out_path}")