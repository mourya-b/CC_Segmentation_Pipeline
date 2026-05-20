#!/bin/bash
#SBATCH --job-name=cc_classifier
#SBATCH --partition=normal
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/data/diag/mouryaBandaru/experiments/classifier_v18b/logs/%j.out
#SBATCH --error=/data/diag/mouryaBandaru/experiments/classifier_v18b/logs/%j.err

mkdir -p /data/diag/mouryaBandaru/experiments/classifier_v18b/logs

# Source paths on Chansey
PECTUS_DICOM_SRC=/data/diag/rubenvdw/Dataset/DICOMS_Pectus
ORANGE_DICOM_SRC=/data/diag/rubenvdw/Dataset/ORANGE_Dicoms
PECTUS_SEG_SRC=/data/diag/mouryaBandaru/data/PECTUS_segmentations
ORANGE_SEG_SRC=/data/diag/mouryaBandaru/data/ORANGE_segmentations

# Scratch destination — node-local fast disk
SCRATCH_DIR=${TMPDIR:-/tmp/oct_scratch}
mkdir -p $SCRATCH_DIR/PECTUS_dicoms $SCRATCH_DIR/ORANGE_dicoms
mkdir -p $SCRATCH_DIR/PECTUS_segs $SCRATCH_DIR/ORANGE_segs

PATIENT_LIST=/data/diag/mouryaBandaru/CC_Segmentation_Pipeline/scripts/patient_list.txt

echo "==> Copying patients from $PATIENT_LIST at $(date)"
echo "==> Scratch dir: $SCRATCH_DIR"
df -h $SCRATCH_DIR

copied_dcm=0; missing_dcm=0
copied_seg=0; missing_seg=0

while IFS= read -r pid; do
    [ -z "$pid" ] && continue
    hospital=$(echo "$pid" | awk -F- '{print $1"-"$2}')

    # DICOM — try PECTUS then ORANGE
    if [ -f "$PECTUS_DICOM_SRC/${pid}.dcm" ]; then
        cp "$PECTUS_DICOM_SRC/${pid}.dcm" "$SCRATCH_DIR/PECTUS_dicoms/" && copied_dcm=$((copied_dcm+1))
    elif [ -f "$ORANGE_DICOM_SRC/${pid}.dcm" ]; then
        cp "$ORANGE_DICOM_SRC/${pid}.dcm" "$SCRATCH_DIR/ORANGE_dicoms/" && copied_dcm=$((copied_dcm+1))
    else
        missing_dcm=$((missing_dcm+1))
    fi

    # Segmentations — try PECTUS then ORANGE
    if [ -d "$PECTUS_SEG_SRC/$hospital/$pid" ]; then
        mkdir -p "$SCRATCH_DIR/PECTUS_segs/$hospital/$pid"
        cp -r "$PECTUS_SEG_SRC/$hospital/$pid/." "$SCRATCH_DIR/PECTUS_segs/$hospital/$pid/" && copied_seg=$((copied_seg+1))
    elif [ -d "$ORANGE_SEG_SRC/$hospital/$pid" ]; then
        mkdir -p "$SCRATCH_DIR/ORANGE_segs/$hospital/$pid"
        cp -r "$ORANGE_SEG_SRC/$hospital/$pid/." "$SCRATCH_DIR/ORANGE_segs/$hospital/$pid/" && copied_seg=$((copied_seg+1))
    else
        missing_seg=$((missing_seg+1))
    fi

done < "$PATIENT_LIST"

echo "==> DICOMs: $copied_dcm copied, $missing_dcm missing"
echo "==> Segs:   $copied_seg copied, $missing_seg missing"
echo "==> Scratch usage:"
du -sh $SCRATCH_DIR
echo "==> Copy complete at $(date)"

export PYTHONPATH=/data/diag/mouryaBandaru/CC_Segmentation_Pipeline

srun --container-image="dockerdex.umcn.nl:5005#mourya-b/cc_segmentation_pipeline:v1.2" \
     --container-mounts="$SCRATCH_DIR:$SCRATCH_DIR,/data/diag:/data/diag" \
     --container-workdir="/data/diag/mouryaBandaru/CC_Segmentation_Pipeline" \
     python src/training/train_classifier.py \
     --config configs/train_classifier_cluster_scratch.yaml