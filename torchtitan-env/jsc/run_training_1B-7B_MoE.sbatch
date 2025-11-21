#!/usr/bin/env bash

# Run a model training on JUWELS Booster.

#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48  # 48/128 physical cores per node.
#SBATCH --threads-per-core=1  # Use only physical CPU cores.
#SBATCH --gres=gpu:4
#SBATCH --time=11:59:59
#SBATCH --account=besl
#SBATCH --partition=booster
#SBATCH --array=1-14%1

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
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
export MASTER_PORT=54123

export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_TIMEOUT=20
# Is slower even with sequence parallelism.
# export CUDA_DEVICE_MAX_CONNECTIONS=1

export DEVICES_PER_NODE=4

export GPUS_PER_REPLICA="$DEVICES_PER_NODE"

export NUM_NODES="$SLURM_JOB_NUM_NODES"
export RDZV_ID="$SLURM_JOB_ID"

# config
model_type=1b-7b_llama3-wo_auxloss-EP1
export CONFIG_FILE=/p/project1/trustllm-eu/filatov1/trustllm/torchtitan/repos/torchtitan/torchtitan/models/MoEllama/train_configs/"$model_type".toml

# LR
export LR=0.125
export EMBED_LR=0.125
export UNEMBED_LR=0.125

# batch & steps
export BATCH_SIZE=12
export GLOBAL_BATCH_SIZE=3072
export STEPS=79473

# wandb
exp_name=1b-7b_llama3
# run_name=test-"$SLURM_JOB_ID"
run_name="$exp_name"-lr-"$LR"-bs-"$GLOBAL_BATCH_SIZE"-steps-"$STEPS"

# logging
export DUMP_FOLDER="$checkpoint_dir"/"$exp_name"/"$run_name"
export WANDB_MODE=offline
export WANDB_PROJECT=debug-run
export WANDB_GROUP="$exp_name"
export WANDB_NAME="$run_name"

srun env -u CUDA_VISIBLE_DEVICES bash "$(get_curr_dir)"/../container_run.sh \
     bash "$training_script"

pop_curr_file
