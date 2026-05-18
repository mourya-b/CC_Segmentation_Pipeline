#!/bin/bash
#SBATCH --job-name=cc_cv
#SBATCH --partition=normal
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --container-image="dockerdex.umcn.nl:5005#mourya-b/cc_segmentation_pipeline:v1.2"
#SBATCH --container-mounts="/data/diag:/data/diag"
#SBATCH --container-workdir="/data/diag/mouryaBandaru/CC_Segmentation_Pipeline"
#SBATCH --output=/data/diag/mouryaBandaru/experiments/classifier_v15/logs/cv_%j.out
#SBATCH --error=/data/diag/mouryaBandaru/experiments/classifier_v15/logs/cv_%j.err

mkdir -p /data/diag/mouryaBandaru/experiments/classifier_v15/logs

export PYTHONPATH=/data/diag/mouryaBandaru/CC_Segmentation_Pipeline
python src/training/cross_validation.py --config configs/train_classifier_cluster.yaml