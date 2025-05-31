#!/bin/bash
set -x

# Distributed training
export NUM_NODES=${NUM_NODES:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
echo "NUM_NODES: $NUM_NODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

export EXP_NAME=${EXP_NAME:-"v_triune"}
export REMOTE_REWARD_JOB_ID=${REMOTE_REWARD_JOB_ID:-"j-xxxxxxxxxx"}

export DATA_TRAIN_FILE=${DATA_TRAIN_FILE:-"[/Orsta-Data-47k/train/train_puzzle_puzzlevqa_2648_x2.parquet, /Orsta-Data-47k/train/train_detection_v3det_4000.parquet]"}
export DATA_TEST_FILE=${DATA_TEST_FILE:-"[/Orsta-Data-47k/test/test_math_megabench_237.parquet, /Orsta-Data-47k/test/test_detection_coco_test_multi_2000.parquet]"}

# Model Architecture and Loss Configuration
export ACTOR_CLIP_RATIO=${ACTOR_CLIP_RATIO:-"0.2"}
export ACTOR_CLIP_RATIO_HIGH=${ACTOR_CLIP_RATIO_HIGH:-"0.28"}
export ACTOR_CLIP_RATIO_LOW=${ACTOR_CLIP_RATIO_LOW:-"0.2"}
export ACTOR_ENTROPY_COEFF=${ACTOR_ENTROPY_COEFF:-"0.000"}
export ACTOR_KL_LOSS_COEFF=${ACTOR_KL_LOSS_COEFF:-"0.001"}
export ACTOR_USE_KL_LOSS=${ACTOR_USE_KL_LOSS:-True}
export ACTOR_KL_LOSS_TYPE=${ACTOR_KL_LOSS_TYPE:-"mse"}
export ACTOR_LOSS_AGG_MODE=${ACTOR_LOSS_AGG_MODE:-"token-mean"}
export ENABLE_DUAL_CLIP=${ENABLE_DUAL_CLIP:-"True"}
export ACTOR_USE_LIGER=${ACTOR_USE_LIGER:-False}

# Model Loading and Checkpointing
export ACTOR_LOAD_PATH=${ACTOR_LOAD_PATH:-"/verl_model/Qwen2.5-VL-7B-Instruct"}
export TRAIN_SAVE_FREQ=${TRAIN_SAVE_FREQ:-"5"}
export TRAIN_SAVE_PATH=${TRAIN_SAVE_PATH:-"/verl_exp"}

# FSDP (Fully Sharded Data Parallel) Configuration
export ACTOR_FSDP_OMT_OFFLOAD=${ACTOR_FSDP_OMT_OFFLOAD:-"False"}
export ACTOR_FSDP_PARAM_OFFLOAD=${ACTOR_FSDP_PARAM_OFFLOAD:-"False"}

# PPO Training Parameters
export ACTOR_PPO_GLOBAL_BSZ=${ACTOR_PPO_GLOBAL_BSZ:-"1024"} # Total number of samples used for a single PPO update
export ACTOR_PPO_MICRO_BSZ=${ACTOR_PPO_MICRO_BSZ:-"16"} # Number of samples in each micro-batch for loss calculation (to prevent OOM)
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-"20480"}
export ALGO_ADV_ESTIMATOR=${ALGO_ADV_ESTIMATOR:-"grpo"}
export ALGO_KL_COEF=${ALGO_KL_COEF:-"0.000"}
export LOG_P_MICRO_BSZ=${LOG_P_MICRO_BSZ:-"32"} # Micro-batch size for log-probability calculation

# Data Configuration
export DATA_TRAIN_BATCH_SIZE=${DATA_TRAIN_BATCH_SIZE:-"1024"} # Number of prompts to generate responses for in each batch
export DATA_TEST_BATCH_SIZE=${DATA_TEST_BATCH_SIZE:-"4096"}
export DATA_MAX_RES_LENGTH=${DATA_MAX_RES_LENGTH:-"2048"} # Maximum length of each generated response
export DATA_FILTER_OVERLONG_PROMPTS=${DATA_FILTER_OVERLONG_PROMPTS:-"False"}
export DATA_IMAGE_KEYWORD=${DATA_IMAGE_KEYWORD:-"images"}
export DATA_MAX_PROMPT_LENGTH=${DATA_MAX_PROMPT_LENGTH:-"8192"}
export DATA_SHUFFLE=${DATA_SHUFFLE:-"True"}
export DATA_NUM_EXAMINE_TRAIN=${DATA_NUM_EXAMINE_TRAIN:-0}
export DATA_NUM_EXAMINE_TEST=${DATA_NUM_EXAMINE_TEST:-0}


# Learning Rate and Optimization
export ACTOR_LR=${ACTOR_LR:-"1e-6"}
export ACTOR_LR_FREEZE=${ACTOR_LR_FREEZE:-"[vit,connector]"} # Can be null or a list (e.g., "['vit', 'connector', 'llm']")
export ACTOR_LR_VIT=${ACTOR_LR_VIT:-$ACTOR_LR}
export ACTOR_LR_CONNECTOR=${ACTOR_LR_CONNECTOR:-$ACTOR_LR}
export ACTOR_LR_LLM=${ACTOR_LR_LLM:-$ACTOR_LR}
export WARMUP_STYLE=${WARMUP_STYLE:-"constant"}
export LR_WARMUP_STEPS_RATIO=${LR_WARMUP_STEPS_RATIO:-"0.05"}

# Rollout Configuration
export ROLLOUT_CHUNKED_PREFILL=${ROLLOUT_CHUNKED_PREFILL:-"False"}
export ROLLOUT_FREE_CACHE_ENFORCE_EAGER=${ROLLOUT_FREE_CACHE_ENFORCE_EAGER:-"False"}
export ROLLOUT_MAX_GPU_MEM=${ROLLOUT_MAX_GPU_MEM:-"0.7"} # Maximum GPU memory to use for rollouts (as a ratio)
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-10240}
export ROLLOUT_N=${ROLLOUT_N:-"8"} # Number of rollout sequences to generate
export ROLLOUT_SWAP_SPACE=${ROLLOUT_SWAP_SPACE:-16} # Swap space in GB for rollouts
export ROLLOUT_TEMP=${ROLLOUT_TEMP:-"1.0"} # Temperature for rollout generation
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-"1"} # Tensor parallelism size for rollout model
export ROLLOUT_IMAGE_LIMIT=${ROLLOUT_IMAGE_LIMIT:-1}
export ROLLOUT_VIDEO_LIMIT=${ROLLOUT_VIDEO_LIMIT:-0}

# Evaluation Configuration
export EVAL_BEFORE_TRAIN=${EVAL_BEFORE_TRAIN:-True}
export EVAL_DO_SAMPLE=${EVAL_DO_SAMPLE:-False}
export EVAL_TEMP=${EVAL_TEMP:-0} # Temperature for evaluation generation
export EVAL_TOPP=${EVAL_TOPP:-1} # Top-p sampling parameter for evaluation

# Training Run Configuration
export TRAIN_PROJECT_NAME=${TRAIN_PROJECT_NAME:-"v_triune"}
export TRAIN_TEST_FREQ=${TRAIN_TEST_FREQ:-"-5"}
export TRAIN_TOTAL_EPOCHS=${TRAIN_TOTAL_EPOCHS:-"3"}
export WANDB_API_KEY=${WANDB_API_KEY:-"your wandb api key"} # Weights & Biases API Key

python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_TRAIN_FILE" \
    data.test_files="$DATA_TEST_FILE" \
    data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
    data.test_batch_size=$DATA_TEST_BATCH_SIZE \
    data.max_prompt_length=$DATA_MAX_PROMPT_LENGTH \
    data.max_response_length=$DATA_MAX_RES_LENGTH \
    data.filter_overlong_prompts=$DATA_FILTER_OVERLONG_PROMPTS \
    data.truncation='error' \
    data.image_key=$DATA_IMAGE_KEYWORD \
    data.shuffle=$DATA_SHUFFLE \
    data.num_examine_train=$DATA_NUM_EXAMINE_TRAIN \
    data.num_examine_test=$DATA_NUM_EXAMINE_TEST \
    actor_rollout_ref.model.path=$ACTOR_LOAD_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=$ACTOR_USE_LIGER \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.optim.lr_vit=$ACTOR_LR_VIT \
    actor_rollout_ref.actor.optim.lr_connector=$ACTOR_LR_CONNECTOR \
    actor_rollout_ref.actor.optim.lr_llm=$ACTOR_LR_LLM \
    actor_rollout_ref.actor.optim.lr_freeze=$ACTOR_LR_FREEZE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ACTOR_PPO_GLOBAL_BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_PPO_MICRO_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.clip_ratio=$ACTOR_CLIP_RATIO \
    actor_rollout_ref.actor.clip_ratio_low=$ACTOR_CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$ACTOR_CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.loss_agg_mode=$ACTOR_LOSS_AGG_MODE \
    actor_rollout_ref.actor.use_kl_loss=$ACTOR_USE_KL_LOSS \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.kl_loss_coef=$ACTOR_KL_LOSS_COEFF \
    actor_rollout_ref.actor.kl_loss_type=$ACTOR_KL_LOSS_TYPE \
    actor_rollout_ref.actor.entropy_coeff=$ACTOR_ENTROPY_COEFF \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$LR_WARMUP_STEPS_RATIO \
    actor_rollout_ref.actor.optim.warmup_style=$WARMUP_STYLE \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_FSDP_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_FSDP_OMT_OFFLOAD \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_P_MICRO_BSZ \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_MAX_GPU_MEM \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMP \
    actor_rollout_ref.rollout.enable_chunked_prefill=$ROLLOUT_CHUNKED_PREFILL \
    actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_FREE_CACHE_ENFORCE_EAGER \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.free_cache_engine=$ROLLOUT_FREE_CACHE_ENFORCE_EAGER \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.engine_kwargs.swap_space=$ROLLOUT_SWAP_SPACE \
    actor_rollout_ref.rollout.val_kwargs.temperature=$EVAL_TEMP \
    actor_rollout_ref.rollout.val_kwargs.top_p=$EVAL_TOPP \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$EVAL_DO_SAMPLE \
    actor_rollout_ref.rollout.limit_images=$ROLLOUT_IMAGE_LIMIT \
    actor_rollout_ref.rollout.limit_videos=$ROLLOUT_VIDEO_LIMIT \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_P_MICRO_BSZ \
    actor_rollout_ref.ref.fsdp_config.param_offload=$ACTOR_FSDP_PARAM_OFFLOAD \
    algorithm.adv_estimator=$ALGO_ADV_ESTIMATOR \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=$ALGO_KL_COEF \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$TRAIN_PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NUM_NODES \
    trainer.default_local_dir=$TRAIN_SAVE_PATH/$_EXP_NAME \
    trainer.save_freq=$TRAIN_SAVE_FREQ \
    trainer.test_freq=$TRAIN_TEST_FREQ \
    trainer.total_epochs=$TRAIN_TOTAL_EPOCHS \
    trainer.resume_mode=auto \
    trainer.val_before_train=$EVAL_BEFORE_TRAIN \
    reward_model.reward_manager=remote \
    +reward_model.remote_reward_job_id=$REMOTE_REWARD_JOB_ID
