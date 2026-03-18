#!/bin/bash
#SBATCH --job-name=cc_classifier
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --container-image="dockerdex.umcn.nl:5005#mourya-b/cc_segmentation_pipeline:v1.0"
#SBATCH --container-mounts="/data/diag:/data/diag"
#SBATCH --output=/data/diag/mouryaBandaru/experiments/classifier_v1/logs/%j.out
#SBATCH --error=/data/diag/mouryaBandaru/experiments/classifier_v1/logs/%j.err

mkdir -p /data/diag/mouryaBandaru/experiments/classifier_v1/logs

cd /data/diag/mouryaBandaru/CC_Segmentation_Pipeline
python src/training/train_classifier.py --config configs/train_classifier_cluster.yaml