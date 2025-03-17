#!/usr/bin/env bash

# Start a training run.

set -euo pipefail

_activated_container="${_ACTIVATED_CONTAINER:-0}"
if ! ((_activated_container)); then
    echo 'Container has not been activated; please use' \
         "\`bash container_run.sh\` to run container scripts."
    exit 1
fi

# -----

torchtitan_repo_dir="$ext_repo_dir"/torchtitan/torchtitan

# Below is a TorchTitan Llama-2 pretraining example configuration,
# with major settings being
# - use variable config values,
# - run multi-node,
# - run for only 10 steps,
# - use BF16 precision,
# - use smaller micro and global batch sizes,
# - use FSDP2 as the sole parallelization strategy,
# - disable gradient accumulation fusion (required for FSDP2),
# - use a local tokenizer,
# - use local preprocessed data from SCRATCH,
# - use multiple CPUs for data processing (variables defined outside
#   script),
# - save checkpoints to SCRATCH,
# - log to SCRATCH.

dataset_files_arg=()
if [ -n "${TRAIN_DATA_FILES:-}" ]; then
    dataset_files_arg=( --training.dataset_files="$TRAIN_DATA_FILES" )
fi

dataset_inner_name_arg=()
if [ -n "${TRAIN_DATA_INNER_NAME:-}" ]; then
    dataset_inner_name_arg=( --training.dataset_inner_name="$TRAIN_DATA_INNER_NAME" )
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -u -m torchrun_jsc \
       --nproc_per_node=gpu \
       --nnodes="$NUM_NODES" \
       --rdzv_id="$RDZV_ID" \
       --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
       --rdzv_backend=c10d \
       "$torchtitan_repo_dir"/train.py  \
       --job.config_file="$CONFIG_FILE" \
       --job.dump_folder="$DUMP_FOLDER" \
       --job.description='training' \
       --job.print_args \
       --metrics.log_freq="$LOG_FREQ" \
       --metrics.log_norm_freq="$LOG_NORM_FREQ" \
       --metrics.enable_tensorboard \
       --metrics.save_tb_folder=logs \
       --metrics.enable_wandb \
       --metrics.wandb_project="$WANDB_PROJECT" \
       --metrics.wandb_group="$WANDB_GROUP" \
       --metrics.wandb_name="$WANDB_NAME" \
       --metrics.rank_0_only \
       --optimizer.name="$OPTIMIZER_NAME" \
       --optimizer.lr="$LR" \
       --optimizer.eps="$EPS" \
       --optimizer.backend_steps="$BACKEND_STEPS" \
       --optimizer.momentum="$MOMENTUM" \
       --optimizer.nesterov \
       --optimizer.embed_lr="$EMBED_LR" \
       --optimizer.unembed_lr="$UNEMBED_LR" \
       --optimizer.embed_str_match="$EMBED_STR_MATCH" \
       --optimizer.unembed_str_match="$UNEMBED_STR_MATCH" \
       --lr_scheduler.warmup_steps="$WARMUP_STEPS" \
       --lr_scheduler.decay_ratio="$DECAY_RATIO" \
       --lr_scheduler.decay_type="$DECAY_TYPE" \
       --lr_scheduler.lr_min="$LR_MIN" \
       --model.name="$MODEL_NAME" \
       --model.flavor="$MODEL_FLAVOR" \
       --model.norm_type="$NORM_TYPE" \
       --model.tokenizer_path="$TOKENIZER_MODEL_FILE" \
       --training.seed="$SEED" \
       --training.dataset=simple_custom \
       --training.dataset_path="$TRAIN_DATA_PATH" \
       "${dataset_files_arg[@]}" \
       "${dataset_inner_name_arg[@]}" \
       --training.dataset_streaming \
       --training.steps="$STEPS" \
       --training.seq_len="$SEQ_LEN" \
       --training.batch_size="$BATCH_SIZE" \
       --training.global_batch_size="$GLOBAL_BATCH_SIZE" \
       --training.max_norm="$MAX_NORM" \
       --training.data_parallel_replicate_degree=1 \
       --training.data_parallel_shard_degree=-1 \
       --training.fsdp_reshard_after_forward=default \
       --training.tensor_parallel_degree=1 \
       --training.compile \
       --training.mixed_precision_param=bfloat16 \
       --training.mixed_precision_reduce=float32 \
       --checkpoint.enable_checkpoint \
       --checkpoint.folder=checkpoints \
       --checkpoint.interval="$CHECKPOINT_INTERVAL" \
       --checkpoint.export_dtype=bfloat16 \
       --checkpoint.async_mode="disabled" \
       --activation_checkpoint.mode="none" \
       --activation_checkpoint.selective_ac_option=op

    #    --training.data_parallel_replicate_degree="$(((NUM_NODES * DEVICES_PER_NODE)))" \
    #    --training.data_parallel_shard_degree=1 \
    #    --training.fsdp_reshard_after_forward=never \
    
    #    --training.data_parallel_replicate_degree="$(((NUM_NODES * DEVICES_PER_NODE) / GPUS_PER_REPLICA))" \
    #    --training.data_parallel_shard_degree=-1 \
    #    --training.fsdp_reshard_after_forward=default \