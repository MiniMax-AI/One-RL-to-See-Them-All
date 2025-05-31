#!/usr/bin/env bash

python scripts/model_merger.py \
  --backend fsdp \
  --hf_model_path /verl_model/Qwen2.5-VL-7B-Instruct \
  --local_dir /verl_exp/your_exp_name/global_step_xxx/actor \
  --target_dir /verl_exp/your_target_dir_name \
