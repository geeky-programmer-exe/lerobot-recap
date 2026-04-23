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
TOTAL_EPISODES=${TOTAL_EPISODES:-5000}
WARMUP_EPISODES=${WARMUP_EPISODES:-200}
EVAL_FREQ=${EVAL_FREQ:-100}
EVAL_EPISODES=${EVAL_EPISODES:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
ACTOR_LR=${ACTOR_LR:-3e-4}
CRITIC_LR=${CRITIC_LR:-3e-4}
GAMMA=${GAMMA:-0.99}
TAU=${TAU:-0.005}
BETA=${BETA:-1.0}
REF_ACTION_DROPOUT_PROB=${REF_ACTION_DROPOUT_PROB:-0.5}
ACTOR_OUTPUT_VARIANCE=${ACTOR_OUTPUT_VARIANCE:-0.05}
Q_LOSS_WEIGHT_MAX=${Q_LOSS_WEIGHT_MAX:-0.5}
Q_LOSS_WEIGHT_INCREMENT=${Q_LOSS_WEIGHT_INCREMENT:-0.05}
Q_CURRICULUM_START_SUCCESS_RATE=${Q_CURRICULUM_START_SUCCESS_RATE:-0.3}
EVAL_VIDEOS_TO_SAVE=${EVAL_VIDEOS_TO_SAVE:-1}
GRAD_UPDATES_PER_TRANSITION=${GRAD_UPDATES_PER_TRANSITION:-1}

python src/lerobot/scripts/train_actor_critic_rlt.py \
  --vla_checkpoint=$OCEAN/checkpoints/peg-sft-c10/checkpoints/last/pretrained_model \
  --rlt_checkpoint=$OCEAN/checkpoints/rlt3/checkpoints/last/pretrained_model \
  --output_dir=$OCEAN/checkpoints/$RUN_NAME \
  --total_episodes=$TOTAL_EPISODES \
  --warmup_episodes=$WARMUP_EPISODES \
  --eval_freq=$EVAL_FREQ \
  --eval_episodes=$EVAL_EPISODES \
  --batch_size=$BATCH_SIZE \
  --actor_lr=$ACTOR_LR \
  --critic_lr=$CRITIC_LR \
  --gamma=$GAMMA \
  --tau=$TAU \
  --beta=$BETA \
  --ref_action_dropout_prob=$REF_ACTION_DROPOUT_PROB \
  --actor_output_variance=$ACTOR_OUTPUT_VARIANCE \
  --q_loss_weight_max=$Q_LOSS_WEIGHT_MAX \
  --q_loss_weight_increment=$Q_LOSS_WEIGHT_INCREMENT \
  --q_curriculum_start_success_rate=$Q_CURRICULUM_START_SUCCESS_RATE \
  --eval_videos_to_save=$EVAL_VIDEOS_TO_SAVE \
  --G=$GRAD_UPDATES_PER_TRANSITION \
  --wandb \
  --wandb_project=rlt-smolvla \
  --wandb_entity=idl_34 \
  --job_name=$RUN_NAME
