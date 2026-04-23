#!/bin/bash
set -euo pipefail

BETAS=(1.0 2.0 3.0)
Q_MAXES=(0.2 0.3)
Q_START_SRS=(0.3 0.5)

for beta in "${BETAS[@]}"; do
  for q_max in "${Q_MAXES[@]}"; do
    for q_start_sr in "${Q_START_SRS[@]}"; do
      job_name=$(printf "rlt-b%s-q%s-s%s" "$beta" "$q_max" "$q_start_sr" | tr '.' 'p')
      echo "Submitting ${job_name}"
      sbatch \
        --job-name "${job_name}" \
        --export=ALL,\
BETA="${beta}",\
Q_LOSS_WEIGHT_MAX="${q_max}",\
Q_LOSS_WEIGHT_INCREMENT="0.05",\
Q_CURRICULUM_START_SUCCESS_RATE="${q_start_sr}",\
REF_ACTION_DROPOUT_PROB="0.5",\
ACTOR_OUTPUT_VARIANCE="0.05",\
GRAD_UPDATES_PER_TRANSITION="1",\
WARMUP_EPISODES="200",\
TOTAL_EPISODES="5000",\
EVAL_FREQ="100",\
EVAL_EPISODES="10",\
EVAL_VIDEOS_TO_SAVE="1",\
BATCH_SIZE="256",\
ACTOR_LR="3e-4",\
CRITIC_LR="3e-4",\
GAMMA="0.99",\
TAU="0.005" \
        scripts/train_rlt_ac.sh
    done
  done
done
