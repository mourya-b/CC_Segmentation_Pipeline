"""
Run once to extract fold val patients from the CV training log.
Saves cv_splits.json into the experiment directory.

Usage:
    python3 scripts/extract_cv_splits.py \
        --log /data/diag/mouryaBandaru/experiments/classifier_v18b/logs/cv_826218.out \
        --out /data/diag/mouryaBandaru/experiments/classifier_v18b/cv_splits.json
"""
import re
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True, help="Path to CV .out log file")
parser.add_argument("--out", required=True, help="Path to save cv_splits.json")
args = parser.parse_args()

text = Path(args.log).read_text()

folds = {}
current_fold = None
for line in text.splitlines():
    fold_match = re.match(r"\s*FOLD\s+(\d+)/\d+", line)
    val_match = re.match(r"\s*Val:\s+(\[.+\])", line)
    if fold_match:
        current_fold = int(fold_match.group(1))
    elif val_match and current_fold is not None:
        folds[current_fold] = eval(val_match.group(1))

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(folds, f, indent=2)

print(f"Wrote {len(folds)} folds to {out_path}")
for fold_num, patients in sorted(folds.items()):
    print(f"  Fold {fold_num}: {len(patients)} val patients — {patients}")