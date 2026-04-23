#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260117p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=40:00:00
#SBATCH --output=/ocean/projects/cis260117p/shared/logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis260117p/shared/logs/%x_%j.err

REPO=/ocean/projects/cis260117p/$USER/lerobot
OCEAN=/ocean/projects/cis260117p/shared

export HF_LEROBOT_HOME=$OCEAN/data
export HF_HOME=$OCEAN/hf_cache
export WANDB_DIR=$OCEAN/wandb
export MUJOCO_GL=osmesa

module load anaconda3
conda activate rlt
cd $REPO

RUN_NAME=${SLURM_JOB_NAME:-rlt-ac}

python src/lerobot/scripts/train_actor_critic_rlt.py \
  --vla_checkpoint=$OCEAN/checkpoints/peg-sft-c10/checkpoints/last/pretrained_model \
  --rlt_checkpoint=$OCEAN/checkpoints/rlt3/checkpoints/last/pretrained_model \
  --output_dir=$OCEAN/checkpoints/$RUN_NAME \
  --total_episodes=5000 \
  --warmup_episodes=200 \
  --eval_freq=100 \
  --eval_episodes=10 \
  --batch_size=256 \
  --actor_lr=3e-4 \
  --critic_lr=3e-4 \
  --gamma=0.99 \
  --tau=0.005 \
  --beta=1.0 \
  --ref_action_dropout_prob=0.5 \
  --actor_output_variance=0.05 \
  --q_loss_weight_max=0.5 \
  --q_loss_weight_increment=0.05 \
  --q_curriculum_start_success_rate=0.3 \
  --eval_videos_to_save=1 \
  --G=1 \
  --wandb \
  --wandb_project=rlt-smolvla \
  --wandb_entity=idl_34 \
  --job_name=$RUN_NAME
