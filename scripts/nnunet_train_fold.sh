#!/bin/bash
#SBATCH --job-name=nnunet_cc_f0
#SBATCH --partition=normal
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/mouryabandaru/nnunet_data/logs/%j_train.out
#SBATCH --error=/home/mouryabandaru/nnunet_data/logs/%j_train.err

# Which fold to train — pass as $1, defaults to 0
FOLD=${1:-0}

mkdir -p /home/mouryabandaru/nnunet_data/logs

srun --container-image="dockerdex.umcn.nl:5005#mourya-b/cc_segmentation_pipeline:v1.2" \
     --container-mounts="/data/diag:/data/diag,/home/mouryabandaru:/home/mouryabandaru" \
     bash -c "
       export nnUNet_raw=/data/diag/mouryaBandaru/nnunet_data/nnUNet_raw
       export nnUNet_preprocessed=/home/mouryabandaru/nnunet_data/nnUNet_preprocessed
       export nnUNet_results=/home/mouryabandaru/nnunet_data/nnUNet_results
       pip install nnunetv2 --quiet && \
       nnUNetv2_train 1 2d $FOLD
     "