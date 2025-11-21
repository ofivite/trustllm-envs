#!/usr/bin/env bash

# Run a model training on JUWELS Booster.

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48  # 48/128 physical cores per node.
#SBATCH --threads-per-core=1  # Use only physical CPU cores.
#SBATCH --gres=gpu:4
#SBATCH --time=03:59:59
#SBATCH --account=besl
# SBATCH --account=generativeaims
# SBATCH --account=westai0000004
# SBATCH --account=training2520
#SBATCH --partition=booster
# SBATCH --partition=dc-gpu
# SBATCH --array=11
#SBATCH --array=3-23:4
# SBATCH --array=2-22:4
# SBATCH --array=4-20:4

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

######################################################

model_type=1b_proxy_llama3_nt_32depth # 1b_proxy_llama3_nt_4width //// 1b_proxy_llama3_nt_8depth
export SEED=303409 # 30 // 3034 // 303409

# config
export CONFIG_FILE=/p/project1/trustllm-eu/filatov1/trustllm/torchtitan/repos/torchtitan/torchtitan/models/llama3/train_configs/"$model_type".toml

# optimizer
export OPTIMIZER_NAME=DistributedScion

# base_lrs=( 0.00006103515 0.00024414 0.00097656 0.00390625 0.015625 0.0625 0.25 1.0 4.0 2.0 8.0 10.0 12.0 16.0)
# base_lrs=( 0.00012207 0.00048828 0.00195312 0.0078125 0.03125 0.125 0.5 1.5 3.0 6.0 11.0 14.0)
base_lrs=( 0.00024414 0.00034527 0.00048828 0.00069053 0.00097656 0.00138107 0.00195312 0.00276214 0.00390625 0.00552427 0.0078125 0.01104854 0.015625 0.02209709 0.03125 0.04419417 0.0625 0.08838835 0.125 0.1767767 0.25 0.35355339 0.5 0.70710678 1.0 1.41421356 2.0 2.82842712 4.0) # np.logspace(-12, 2, 29, base=2)
# base_lrs=( 7.62939453e-06 1.07895932e-05 1.52587891e-05 2.15791864e-05 3.05175781e-05 4.31583729e-05 6.10351562e-05 8.63167458e-05 1.22070312e-04 1.72633492e-04 2.44140625e-04) # np.logspace(-17, -12, 11, base=2)

# emb_base_lrs=( 1.e-05 1.e-04 1.e-03 1.e-02 1.e-01 1.e+00 )
# unemb_base_lrs=( 1.e-05 1.e-04 1.e-03 1.e-02 1.e-01 1.e+00 )
# emb_base_lrs=( 9.765625e-03 1.953125e-02 3.906250e-02 7.812500e-02 1.562500e-01 3.125000e-01 6.250000e-01 1.250000e+00 2.500000e+00 5.000000e+00 1.000000e+01 2.000000e+01 )
# unemb_base_lrs=( 9.765625e-03 1.953125e-02 3.906250e-02 7.812500e-02 1.562500e-01 3.125000e-01 6.250000e-01 1.250000e+00 2.500000e+00 5.000000e+00 1.000000e+01 2.000000e+01 )

embed_lr_id=$SLURM_ARRAY_TASK_ID
hidden_lr_id=$SLURM_ARRAY_TASK_ID
unembed_lr_id=$SLURM_ARRAY_TASK_ID

export LR="${base_lrs[$hidden_lr_id]}"
export EMBED_LR="${base_lrs[$embed_lr_id]}"
export UNEMBED_LR="${base_lrs[$unembed_lr_id]}"

export DECAY_RATIO=0.0
export LR_MIN=1.0

export WEIGHT_DECAY=0.0
export MOMENTUM=1.0

# batch & steps
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=1024
export STEPS=2048

# checkpointing / logging
export LOAD_STEP=-1
export CHECKPOINT_INTERVAL=512
export LOG_FREQ=16
export LOG_NORM_FREQ=128

# wandb
exp_name=eta-BS-scan-x32-depth
run_name=lr-"$LR"-bs-"$GLOBAL_BATCH_SIZE"-seed-"$SEED"
# run_name=lr-i"$embed_lr_id"h"$hidden_lr_id"o"$unembed_lr_id"-bs-"$GLOBAL_BATCH_SIZE"-seed-"$SEED"
# run_name=lr-"$LR"-bs-"$GLOBAL_BATCH_SIZE"-momentum-"$MOMENTUM"-wd-"$WEIGHT_DECAY"-LOG_FREQ-"$LOG_FREQ"
# run_name=test-"$SLURM_JOB_ID"-"$model_type"-mbs-"$BATCH_SIZE"-nnodes-"$SLURM_JOB_NUM_NODES"


# logging
export DUMP_FOLDER="$checkpoint_dir"/"$exp_name"/"$run_name"
# export CHECKPOINT_FOLDER="$checkpoint_dir"/"$exp_name"/lr-"$LR"-bs-"$GLOBAL_BATCH_SIZE"-momentum-"$MOMENTUM"/checkpoint
export CHECKPOINT_FOLDER=$DUMP_FOLDER
export WANDB_MODE=offline
export WANDB_PROJECT=norm-transfer
export WANDB_GROUP="$exp_name"
export WANDB_NAME="$run_name"

srun env -u CUDA_VISIBLE_DEVICES bash "$(get_curr_dir)"/../container_run.sh \
     bash "$training_script"

pop_curr_file
