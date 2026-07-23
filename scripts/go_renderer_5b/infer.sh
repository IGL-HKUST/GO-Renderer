#!/bin/bash
set -x

# Model and example
MODEL_PATH="models/Diffusion_Transformer/GO-Renderer-5B"
VALIDATION_JSON="assets/validation/metadata.json"
OUTPUT_DIR="outputs/go_renderer_5b/crystalpig"
GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/go_renderer_5b/infer.py \
  --config_path config/wan2.2/wan_civitai_5b.yaml \
  --model_name "$MODEL_PATH" \
  --validation_json "$VALIDATION_JSON" \
  --validation_samples 1 \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --num_ref_frames 8 \
  --fps 16 \
  --num_inference_steps 25 \
  --sampler_name Flow_Unipc \
  --shift 5 \
  --output_dir "$OUTPUT_DIR" \
  --gpu_memory_mode model_full_load \
  --weight_dtype bfloat16
