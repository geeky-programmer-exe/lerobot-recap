#!/bin/bash
#SBATCH --job-name=zrl-pca
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260117p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=20:00:00
#SBATCH --output=/ocean/projects/cis260117p/shared/logs/zrl-pca_%j.out
#SBATCH --error=/ocean/projects/cis260117p/shared/logs/zrl-pca_%j.err

REPO=/ocean/projects/cis260117p/$USER/lerobot
OCEAN=/ocean/projects/cis260117p/shared
HF_HUB_OFFLINE=1

export HF_LEROBOT_HOME=$OCEAN/data
export HF_HOME=$OCEAN/hf_cache
export WANDB_DIR=$OCEAN/wandb
export MUJOCO_GL=osmesa

module load anaconda3
conda activate rlt
cd $REPO

python src/lerobot/scripts/eval_zrl_pca.py \
    --vla_checkpoint=$OCEAN/checkpoints/peg-sft-c10/checkpoints/last/pretrained_model \
    --rlt_checkpoint=$OCEAN/checkpoints/rlt-no-x-attn/checkpoints/last/pretrained_model  \
    --n_episodes 30 \
    --perturb_std 0.02 \
    --max_videos 5 \
    --output_dir ./zrl_pca_out