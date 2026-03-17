#!/bin/bash
#SBATCH --job-name=cc_classifier
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/data/diag/mouryaBandaru/experiments/classifier_v1/logs/%j.out
#SBATCH --error=/data/diag/mouryaBandaru/experiments/classifier_v1/logs/%j.err

# Create output/log dirs if they don't exist
mkdir -p /data/diag/mouryaBandaru/experiments/classifier_v1/logs

# Pull latest image
docker pull dockerdex.umcn.nl:5005/mourya-b/cc_segmentation_pipeline:v1.0

# Run training
docker run --rm --gpus all \
    -v /data/diag:/data/diag \
    dockerdex.umcn.nl:5005/mourya-b/cc_segmentation_pipeline:v1.0 \
    python src/training/train_classifier.py --config configs/train_classifier_cluster.yaml