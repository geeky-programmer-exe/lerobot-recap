#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260117p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=20:00:00
#SBATCH --output=/ocean/projects/cis260117p/shared/logs/peg-sft_%j.out
#SBATCH --error=/ocean/projects/cis260117p/shared/logs/peg-sft_%j.err

REPO=/ocean/projects/cis260117p/$USER/lerobot
OCEAN=/ocean/projects/cis260117p/shared
HF_HUB_OFFLINE=1

module load anaconda3
conda activate rlt
cd $REPO

python src/lerobot/scripts/eval_patched_vla.py \
  --vla_checkpoint=$OCEAN/checkpoints/peg-sft-c10/checkpoints/last/pretrained_model \
  --rlt_checkpoint=$OCEAN/checkpoints/rlt3/checkpoints/last/pretrained_model  \
  --ac_checkpoint=$OCEAN/checkpoints/rlt-ac-sparse-b0.5/checkpoint_ep900.pt \
  --n_episodes 50 \
  # --video_dir ./eval_videos \
  # --max_videos 10
