#!/usr/bin/env bash

# Run a model training on JUWELS Booster.

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48  # 48 physical cores per node.
#SBATCH --threads-per-core=1  # Use only physical CPU cores.
#SBATCH --gres=gpu:4
#SBATCH --time=05:59:59
#SBATCH --account=besl
# Use `develbooster` for debugging, `booster` for "normal" jobs, and
# `largebooster` for jobs on more than 256 nodes.
#SBATCH --partition=booster
#SBATCH --array=4,5,7
# SBATCH --exclude=jwb[0150,0812,0392,0268,0796,0713,0716,0726,1085,0363,0477,0898,1004,0063,0824,0199,0232,1213,0801,0547,0900,0361,0646,0059,0294,0927,0502,0645,1239,0675,0634,0287,0296,0971,0096,1064,0454,1125,1023,1245,0036,0227,1247,0186,1047,0449,1246,1244,0193,1057,0474,0895,1117,0381,1240,0935,0130,1215,0476,0213,0058,1243,1214,0542,0266,0026,1156,0607,0075,0021,0418,1126,1067,0359,0253,1242,0748,0149,0550,0391,1029,0680,1248,1241,1218,1116,1022,0712,0617,0201,0127,1238,1227,1225,1224,1223,1221,1219,1217,0676,0262,0650,0513,0514,0515,0516,0517,0518,0519,0520,0521,0522,0523,0524,0533,0534,0535,0536,0537,0538,0539,0540,0541,0543,0544,0545,0546,0549,0551,0552,0553,0555,0556,0565,0566,0567,0568,0569,0570,0571,0572,0573,0575,1025,1027,1030,1031,1032,1034,1035,1036,1045,1048,1049,1050,1051,1052,1054,1055,1056,1058,1060,1068,1077,1083,1086,0131-0134,0136-0140,0151-0157,0159-0172,0181-0185,0187-0192,0769-0780,0789-0795,0797-0800,0262,0231,0418,0086,0089-0090,0100,0102-0104,0106-0107,0118-0120,0122-0123,0125,0128,0620,0629-0633,0636-0637,0639,0707-0708,0710-0711,0714,0725,0727-0729,0732,0734-0735,0737,0739-0741,0744-0745,0747,0757-0763,0768,0802,0804-0808,0810-0811,0821-0823,0825,0407,0003,0035,0037-0038,0041,0057,0061,0064,0071-0074,0087,0091-0095,0098-0099,0124,0455-0456,0899,0901-0904,0919-0920,0928,0938,0002,0004-0007,0023-0025,0031-0032,0581,0608,0673-0674,0677-0678,0684,0693-0694,0696-0697,0701-0703,0987-0989,0992,0997,1014-1016]
# SBATCH --nodelist=jwb[0257-0260,0278-0281]

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

export DEVICES_PER_NODE=4

# # NCCL timeout 
# export NCCL_LL_THRESHOLD=0
# export NCCL_IB_TIMEOUT=500
# export UCX_RC_TIMEOUT=40s
# export NCCL_IB_RETRY_CNT=100

# export UCX_DC_MLX5_FC_ENABLE=y
# export UCX_DC_MLX5_TIMEOUT=10000000.00us
# export UCX_DC_MLX5_RNR_TIMEOUT=10000.00us
# export UCX_RC_MLX5_FC_ENABLE=y
# export UCX_RC_MLX5_TIMEOUT=10000000.00us
# export UCX_RC_MLX5_RNR_TIMEOUT=10000.00us
# export UCX_UD_MLX5_FC_ENABLE=y
# export UCX_UD_MLX5_TIMEOUT=10000000.00us
# export UCX_UD_MLX5_RNR_TIMEOUT=10000.00us

# We have 48 physical CPU cores on JUWELS Booster nodes, so configure
# 8 + 3 = 11 data workers per GPU in total; this leaves one CPU for
# the main process.
export TRAIN_NUM_WORKERS=8
export EVAL_NUM_WORKERS=3

# model cfg
export TRAIN_CONFIG_YAML_FILE=train/yamls/pretrain/mpt-1b.yaml

# data
export INPUT_DATA_ROOT_DIR="$data_dir"/baseline-llm-foundry

# random seed
export GLOBAL_SEED=1012 #  // 111

# tokenizer
export MAX_SEQ_LEN=4096
export TOKENIZER_DIR=/p/data1/trustllmd/filatov1/tokenizers/baseline-tok

# model
export N_LAYERS=2
export D_MODEL=4096
export D_HEAD=64
export EXPANSION_RATIO=4

export KV_N_HEADS_BASE=2
export D_MODEL_BASE=256
# export N_HEADS_BASE="$((D_MODEL_BASE / D_HEAD))"
export N_HEADS="$((D_MODEL / D_HEAD))"
export KV_N_HEADS="$((KV_N_HEADS_BASE * D_MODEL / D_MODEL_BASE))"

# optimizer
export OPTIMIZER_NAME=muon
# export BETA_1=0.9
# export BETA_2=0.98
# export WEIGHT_DECAY=0 # 1e-02
export EPS_BASE=1e-10
base_lrs=( 0.00024414 0.00048828 0.00097656 0.00195312 0.00390625 0.0078125 0.015625 0.03125 0.0625 ) # np.logspace(-12, -4, 9, base=2)
export LR_BASE="${base_lrs[$SLURM_ARRAY_TASK_ID]}"
export EMBEDDING_LR="${base_lrs[$SLURM_ARRAY_TASK_ID]}"
export EMBEDDING_MOMENTUM=0.95
export EMBEDDING_NESTEROV=true
export EMBEDDING_DUAL=false
# export INIT_STD_BASE=0.04419417382 # "$((1 / D_MODEL**0.5))"
export DUAL_NORM_SCALING=false

# batching
export MICRO_BS=1
export EVAL_MICRO_BS="$((1 * MICRO_BS))"
export GLOBAL_BS=256

# parallelisation / performance
export PRECISION="amp_bf16" # amp_bf16 / amp_fp16
# export FSDP_STRATEGY="FULL_SHARD"
export ACCUMULATE_TRAIN_BATCH_ON_TOKENS=true # Whether training loss is accumulated over the number of tokens in a batch rather than the number of samples. Default: true

# schedule
export SCHEDULER_NAME="constant_decay_with_warmup"
export T_WARMUP="0ba"
export T_CONSTANT="6144ba"
export T_DECAY="2048ba"
export ALPHA_F=0.0
export MAX_DURATION="8192ba"
# export RESET_TIME=false

# eval / checkpointing intervals
export EVAL_INTERVAL="64ba" # "1600ba"
export SAVE_INTERVAL="256ba"
export SAVE_OVERWRITE=true
export SAVE_NUM_CHECKPOINTS_TO_KEEP=1

# experiment
export EXPERIMENT_NAME=muon-scaling-v1-width-"$D_MODEL"
# export RUN_NAME=run-width-"$D_MODEL"-lr-"$SLURM_ARRAY_TASK_ID"-bs-"$GLOBAL_BS"-embed-lr-"$EMBEDDING_LR"-mom-"$EMBEDDING_MOMENTUM"-eps-"$EPS_BASE"-dual-scaling-"$DUAL_NORM_SCALING"-emb-dual-"$EMBEDDING_DUAL"-opt-v2-init-v3
export RUN_NAME=run-lr-"$SLURM_ARRAY_TASK_ID"-bs-"$GLOBAL_BS"-dur-"$MAX_DURATION"-embed-lr-"$EMBEDDING_LR"-mom-"$EMBEDDING_MOMENTUM"-eps-"$EPS_BASE"-dual-scaling-"$DUAL_NORM_SCALING"-opt-v1-init-v3-no-clip #-seed-"$GLOBAL_SEED"

# save/load paths
export SAVE_FOLDER="$checkpoint_dir"/"$EXPERIMENT_NAME"/"$RUN_NAME"/checkpoints
export TENSORBOARD_LOG_DIR="$checkpoint_dir"/"$EXPERIMENT_NAME"/"$RUN_NAME"/tensorboard_logs
export MLFLOW_LOG_DIR="$checkpoint_dir"/mlruns
export MLFLOW_RESUME=false
export LOAD_PATH=null
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
