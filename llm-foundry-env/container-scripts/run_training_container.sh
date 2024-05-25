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

# Set defaults for number of workers if not given.
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-0}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"

cd "$ext_repo_dir"/llm-foundry/scripts

# Below is the LLM Foundry README quickstart example, modified to
# - not buffer output,
# - run multi-node,
# - use a variable YAML file,
# - use local preprocessed data from SCRATCH,
# - use a local tokenizer,
# - use the name of the dataset as splits,
# - use multiple CPUs for data processing (variables defined outside
#   script),
# - use FlashAttention-2,
# - save checkpoints to SCRATCH.

# Train a model for 10 batches
python -u -m composer \
    --nproc="$DEVICES_PER_NODE" \
    --world_size="$WORLD_SIZE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train/train.py \
    "$TRAIN_CONFIG_YAML_FILE" \
    data_local="$INPUT_DATA_ROOT_DIR" \
    train_loader.num_workers="$TRAIN_NUM_WORKERS" \
    train_loader.dataset.split=train \
    eval_loader.num_workers="$EVAL_NUM_WORKERS" \
    eval_loader.dataset.split=validation \
    global_train_batch_size="$GLOBAL_BS" \
    device_train_microbatch_size="$MICRO_BS" \
    device_eval_batch_size="$EVAL_MICRO_BS" \
    model.d_model="$D_MODEL" \
    model.n_layers="$N_LAYERS" \
    model.n_heads="$N_HEADS" \
    model._mup_config.d_model_base="$D_MODEL_BASE" \
    model._mup_config.n_heads_base="$N_HEADS_BASE" \
    experiment_name="$EXPERIMENT_NAME" \
    run_name="$RUN_NAME" \
    global_seed="$GLOBAL_SEED" \
    precision="$PRECISION" \
    max_duration="$MAX_DURATION" \
    reset_time="$RESET_TIME" \
    optimizer.name="$OPTIMIZER_NAME" \
    optimizer.weight_decay="$WEIGHT_DECAY" \
    scheduler.name="$SCHEDULER_NAME" \
    scheduler.t_warmup="$WARMUP" \
    load_path="$LOAD_PATH" \
    save_folder="$SAVE_FOLDER" \
    save_interval="$SAVE_INTERVAL" \
    save_overwrite="$SAVE_OVERWRITE" \
    save_num_checkpoints_to_keep="$SAVE_NUM_CHECKPOINTS_TO_KEEP" \
    eval_interval="$EVAL_INTERVAL" \
    loggers.wandb.entity="$WANDB_ENTITY" \
    loggers.wandb.project="$WANDB_PROJECT" \
    fsdp_config.sharding_strategy="$FSDP_STRATEGY"

    # scheduler.alpha_f="$ALPHA_F" \

# # Convert the model to HuggingFace format
# python inference/convert_composer_to_hf.py \
#   --composer_path mpt-125m/ep0-ba10-rank0.pt \
#   --hf_output_path mpt-125m-hf \
#   --output_precision bf16 \
#   # --hf_repo_for_upload user-org/repo-name

# # Evaluate the model on a subset of tasks
# composer eval/eval.py \
#   eval/yamls/hf_eval.yaml \
#   icl_tasks=eval/yamls/copa.yaml \
#   model_name_or_path=mpt-125m-hf

# # Generate responses to prompts
# python inference/hf_generate.py \
#   --name_or_path mpt-125m-hf \
#   --max_new_tokens 256 \
#   --prompts \
#     "The answer to life, the universe, and happiness is" \
#     "Here's a quick recipe for baking chocolate chip cookies: Start by"
