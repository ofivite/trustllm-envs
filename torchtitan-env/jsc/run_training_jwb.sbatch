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
#SBATCH --array=9

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
export CONFIG_FILE=/p/project1/trustllm-eu/filatov1/trustllm/torchtitan/repos/torchtitan/torchtitan/models/llama3/train_configs/dev_normy.toml

# model specification
model_kind=1
export SEED=123
export MODEL_NAME=byte_llama3 # bit_byte_llama3 / byte_MoEllama3
export MODEL_FLAVOR=debugmodel-multiplier-"$model_kind" # debugmodel-multiplier-"$model_kind"
export NORM_TYPE=np_rmsnorm

# optimizer
#### (for Scion) make sure setting is_light / is_unconstrained / nesterov in `toml` file 
export OPTIMIZER_NAME=Scion
base_lrs=( 9.765625e-04 1.953125e-03 3.906250e-03 7.812500e-03 1.562500e-02 3.125000e-02 6.250000e-02 1.250000e-01 2.500000e-01 5.000000e-01 1.000000e+00 2.000000e+00 ) # np.logspace(-10, 1, 12, base=2)
# export OPTIMIZER_NAME=AdamW
# base_lrs=( 3.05175781e-05 6.10351562e-05 1.22070312e-04 2.44140625e-04 4.88281250e-04 9.76562500e-04 1.95312500e-03 3.90625000e-03 7.81250000e-03 1.56250000e-02 3.12500000e-02) # np.logspace(-15, -5, 11, base=2)
export LR="${base_lrs[$SLURM_ARRAY_TASK_ID]}"
export EPS=1e-12
export BACKEND_STEPS=5
export MOMENTUM=0.1
export EMBED_LR=$LR
export UNEMBED_LR=$LR
export EMBED_STR_MATCH="tok_embeddings.weight"
export UNEMBED_STR_MATCH="output.weight"
export MAX_NORM=-1

# batch
export SEQ_LEN=4096
export BATCH_SIZE=32 # 32 / 16 / 8 / 4 / 2
export GLOBAL_BATCH_SIZE=256

# schedule
export STEPS=4096
export WARMUP_STEPS=0
export DECAY_RATIO=0.25
export DECAY_TYPE=linear
export LR_MIN=0.0

# checkpointing
export CHECKPOINT_INTERVAL=1024

# wandb
exp_name=norm-check-llama-"$OPTIMIZER_NAME"-v1
# run_name=test-"$SLURM_JOB_ID"
run_name=width-"$model_kind"-lr-"$SLURM_ARRAY_TASK_ID"-16layers

# logging
export LOG_FREQ=1
export LOG_NORM_FREQ=64
export DUMP_FOLDER="$checkpoint_dir"/"$exp_name"/"$run_name"
export WANDB_MODE=offline
export WANDB_PROJECT=scaling-titans
export WANDB_GROUP="$exp_name"
export WANDB_NAME="$run_name"

srun env -u CUDA_VISIBLE_DEVICES bash "$(get_curr_dir)"/../container_run.sh \
     bash "$training_script"

pop_curr_file
