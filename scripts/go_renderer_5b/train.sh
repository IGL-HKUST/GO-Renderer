#!/bin/bash
set -x

# GPU configuration
GPU_IDS="0,1,2,3,4,5,6,7"
NUM_PROCESSES=8
PORT=29500

# Training dataset. Edit these two paths before training.
TRAIN_DATA_DIR="/path/to/training/data"
TRAIN_DATA_META="/path/to/training.json"

# Model and output
MODEL_PATH="models/Diffusion_Transformer/GO-Renderer-5B"
OUTPUT_DIR="outputs/go_renderer_5b_train"

# CrystalPig validation uses assets/validation/crystalpig.
CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port "$PORT" \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file config/zero_stage2_config.json \
  scripts/go_renderer_5b/train.py \
  --config_path config/wan2.2/wan_civitai_5b.yaml \
  --model_name "$MODEL_PATH" \
  --train_data_dir "$TRAIN_DATA_DIR" \
  --train_data_meta "$TRAIN_DATA_META" \
  --output_dir "$OUTPUT_DIR" \
  --fix_sample_size 480 832 \
  --image_sample_size 480 \
  --video_sample_size 480 \
  --video_sample_stride 1 \
  --video_sample_n_frames 81 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 20000 \
  --checkpointing_steps 500 \
  --checkpoints_total_limit 5 \
  --learning_rate 2e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 200 \
  --seed 42 \
  --mixed_precision bf16 \
  --adam_weight_decay 3e-2 \
  --adam_epsilon 1e-10 \
  --vae_mini_batch 1 \
  --max_grad_norm 0.05 \
  --dataloader_num_workers 0 \
  --enable_bucket \
  --uniform_sampling \
  --train_mode control_ref \
  --trainable_modules . \
  --report_to tensorboard \
  --tracker_project_name go-renderer-5b \
  --gradient_checkpointing \
  --use_deepspeed \
  --ref_frames_min 3 \
  --ref_frames_max 8 \
  --coordmap_aug_prob 0.1 \
  --coordmap_aug_scale 0.01 \
  --validation_steps 2000 \
  --validation_num_inference_steps 25 \
  --validation_prompts "a small, adorable, clear glass figurine resembling a pig is placed on a desk outside. The background is wonderful dusk scenery with trees and a lake. The camera is moving." \
  --validation_ref_path assets/validation/crystalpig/ref.mp4 \
  --validation_ref_coordmap_path assets/validation/crystalpig/ref_coordmap.mp4 \
  --validation_fg_coordmap_path assets/validation/crystalpig/fg_coordmap.mp4 \
  --validation_size 480 832 81
