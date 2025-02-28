#!/usr/bin/env bash

# Run a model training on JUWELS Booster.

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128  # 48 physical cores per node.
#SBATCH --threads-per-core=1  # Use only physical CPU cores.
#SBATCH --gres=gpu:4
#SBATCH --time=00:04:59
#SBATCH --account=eats-rna
# Use `develbooster` for debugging, `booster` for "normal" jobs, and
# `largebooster` for jobs on more than 256 nodes.
#SBATCH --partition=dc-gpu-devel

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

# # We have 48 physical CPU cores on JUWELS Booster nodes, so configure
# # 11 active data workers per GPU in total; this leaves one CPU for the
# # main process.
# export TRAIN_NUM_WORKERS=11

# # If using TorchTitan's native dataloader, can be a path to a
# # directory, or a file format supported by HuggingFace `datasets`
# # (e.g., the literal "json"). If using the experimental dataloader, a
# # root directory containing directories containing data files has to
# # be given.
# export TRAIN_DATA_PATH=json
# # Optional comma-separated list of files to select. Set as empty
# # string ('') to disable. Ignored when using experimental dataloader.
# export TRAIN_DATA_FILES="$root_scratch_dir"/"$project_name"/example-data/tiny-c4-100k.jsonl
# # Optional name of sub-dataset to use from root dataset (in this case,
# # the "en" subset of C4). Set as empty string ('') to disable. Ignored
# # when using experimental dataloader.
# export TRAIN_DATA_INNER_NAME=en
# # Optional comma-separated list of names of sub-directories in
# # `TRAIN_DATA_PATH` to select for inclusion. Set as empty string ('')
# # to disable. Ignored when using native dataloader.
# # export TRAIN_DATASETS=train
# # We simply use the validation set for both evaluation and testing in
# # the example. Of course, this is redundant, but makes it easier to
# # change the example to include a real test set. Also, we need to
# # supply a test set assuming the data argument is given like in the
# # example.
# # export EVAL_DATA_PATH="$TRAIN_DATA_PATH"
# # export TEST_DATA_PATH="$EVAL_DATA_PATH"

export TRAIN_DATA_PATH=/p/scratch/trustllm-eu/ebert1/torchtitan/data/fineweb-2/deu/train-shuffled_spark_64-nodes
export TRAIN_DATA_FILES=''
export TRAIN_DATA_INNER_NAME=''

export TOKENIZER_MODEL_FILE=TICKIETACKIE
export CONFIG_FILE=/p/project1/trustllm-eu/filatov1/trustllm/torchtitan/repos/torchtitan/torchtitan/models/llama/train_configs/dev_normy.toml

exp_name=llama3_debug
# run_name=test-13493474
run_name=test-"$SLURM_JOB_ID"

# model specification
export SEED=1
export MODEL_NAME=bit_byte_llama3
export MODEL_FLAVOR=debugmodel
export NORM_TYPE=rmsnorm

# optimizer
export OPTIMIZER_NAME=AdamW
export LR=3e-4
export MAX_NORM=1

# training / data schedule
export STEPS=10
export SEQ_LEN=4096
export BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=8
export WARMUP_STEPS=100

# checkpointing
export CHECKPOINT_INTERVAL=1000

# logging
export LOG_FREQ=1
export DUMP_FOLDER="$checkpoint_dir"/"$exp_name"/"$run_name"
export WANDB_MODE=offline
export WANDB_PROJECT=scaling-titans
export WANDB_GROUP="$exp_name"
export WANDB_NAME="$run_name"

srun env -u CUDA_VISIBLE_DEVICES bash "$(get_curr_dir)"/../container_run.sh \
     bash "$training_script"

pop_curr_file
