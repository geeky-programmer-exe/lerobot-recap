#!/bin/bash
#SBATCH --job-name=peg-sft
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260117p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=20:00:00
#SBATCH --output=/ocean/projects/cis260117p/shared/logs/peg-sft_%j.out
#SBATCH --error=/ocean/projects/cis260117p/shared/logs/peg-sft_%j.err

python eval_patched_vla.py \
  --vla_checkpoint=$OCEAN/checkpoints/peg-sft-c10/checkpoints/last/pretrained_model \
  --rlt_checkpoint=$OCEAN/checkpoints/rlt3/checkpoints/last/pretrained_model  \
  --n_episodes 20
