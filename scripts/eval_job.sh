#!/bin/bash
#SBATCH --job-name=cc_eval
#SBATCH --partition=normal
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --container-image="dockerdex.umcn.nl:5005#mourya-b/cc_segmentation_pipeline:v1.2"
#SBATCH --container-mounts="/data/diag:/data/diag"
#SBATCH --container-workdir="/data/diag/mouryaBandaru/CC_Segmentation_Pipeline"
#SBATCH --output=/data/diag/mouryaBandaru/experiments/classifier_v16/logs/eval_%j.out
#SBATCH --error=/data/diag/mouryaBandaru/experiments/classifier_v16/logs/eval_%j.err

export PYTHONPATH=/data/diag/mouryaBandaru/CC_Segmentation_Pipeline
python -m src.evaluation.evaluate_classifier \
  --config configs/train_classifier_cluster.yaml \
  --checkpoint /data/diag/mouryaBandaru/experiments/classifier_v16/best_classifier.pth