#!/usr/bin/env bash

# Run a model training on JUWELS Booster.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48  # 48 physical cores per node.
# SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread  # Use only physical CPU cores.
#SBATCH --gres=gpu:4
#SBATCH --time 00:20:00
#SBATCH --account=atmlaml
# booster / develbooster 
# dc-gpu / dc-gpu-devel
# gpus / develgpus
#SBATCH --partition=dc-gpu-devel
#SBATCH --array=0

set -euo pipefail

# Do not use these variables; they may be overwritten. Instead, use
# `get_curr_file` or `get_curr_dir` after sourcing `get_curr_file.sh`.
_curr_file="$(scontrol show job "$SLURM_JOB_ID" | grep '^[[:space:]]*Command=' | head -n 1 | cut -d '=' -f 2-)"
_curr_dir="$(dirname "$_curr_file")"
source "$_curr_dir"/../../global-scripts/get_curr_file.sh "$_curr_file"

source "$(get_curr_dir)"/../configuration.sh

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
export MASTER_PORT=54123

export DEVICES_PER_NODE=4

# We have 48 physical CPU cores on JUWELS Booster nodes, so configure
# 8 + 3 = 11 data workers per GPU in total; this leaves one CPU for
# the main process.
export TRAIN_NUM_WORKERS=8
export EVAL_NUM_WORKERS=3

# NCCL timeout 
export NCCL_IB_TIMEOUT=500
export UCX_RC_TIMEOUT=40s
export NCCL_IB_RETRY_CNT=100

export UCX_DC_MLX5_FC_ENABLE=y
export UCX_DC_MLX5_TIMEOUT=10000000.00us
export UCX_DC_MLX5_RNR_TIMEOUT=10000.00us
export UCX_RC_MLX5_FC_ENABLE=y
export UCX_RC_MLX5_TIMEOUT=10000000.00us
export UCX_RC_MLX5_RNR_TIMEOUT=10000.00us
export UCX_UD_MLX5_FC_ENABLE=y
export UCX_UD_MLX5_TIMEOUT=10000000.00us
export UCX_UD_MLX5_RNR_TIMEOUT=10000.00us

# random seed
export GLOBAL_SEED=5120 # 5120 (baseline) / 8 (1) / 3 (2)

export TRAIN_CONFIG_YAML_FILE=train/yamls/pretrain/mpt-350m.yaml
export INPUT_DATA_ROOT_DIR="$data_dir"/c4-en-tokenizer-gpt2-concat-1024

# model
export N_LAYERS=24
export D_MODEL=1024
export D_MODEL_BASE=1024
export D_HEAD=128
export N_HEADS="$((D_MODEL / D_HEAD))"
export N_HEADS_BASE="$((D_MODEL_BASE / D_HEAD))"

# machine dependent (juwels/jureca)
export PRECISION="amp_bf16" # amp_bf16 / amp_fp16
export MICRO_BS=16 # max for baseline: 16 (JURECA)
export EVAL_MICRO_BS="$((2 * MICRO_BS))" # 16 is max for 4 layers, 10240 width

# schedule
export SCHEDULER_NAME="constant_with_warmup"  # linear_decay_with_warmup / constant_with_warmup / cosine_with_warmup
export GLOBAL_BS=1024
export WARMUP="512ba" # "10_000_000tok" / "100ba"
export MAX_DURATION="16384ba"
# export ALPHA_F=0.1

# eval / checkpointing intervals
export EVAL_INTERVAL="512ba" # "512ba"
export SAVE_INTERVAL="512ba" # "512ba"
export SAVE_OVERWRITE=false # true / false
export SAVE_NUM_CHECKPOINTS_TO_KEEP=-1 # -1 / 1

# experiment
export EXPERIMENT_NAME=warmup-constant # warmup-constant / warmup-linear / warmup-cosine
export RUN_NAME=lr-"$SLURM_ARRAY_TASK_ID"-bs-"$GLOBAL_BS"-warmup-"$WARMUP"-dur-"$MAX_DURATION" # -job-"$SLURM_JOB_ID"
export SAVE_FOLDER="$checkpoint_dir"/"$EXPERIMENT_NAME"/"$RUN_NAME"/checkpoints

export LOAD_PATH=
# export LOAD_PATH="$SAVE_FOLDER"/latest-rank0.pt

# We assign to a variable again so Bash can do the quoted
# interpolation.
_curr_dir="$(get_curr_dir)"

srun bash -c "
    export WORLD_SIZE=\"\$((SLURM_JOB_NUM_NODES * DEVICES_PER_NODE))\"; \\
    export NODE_RANK=\"\$SLURM_NODEID\"; \\
    bash ${_curr_dir@Q}/../container_run.sh \\
        bash ${training_script@Q}
"

pop_curr_file
