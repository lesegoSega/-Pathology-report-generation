#!/bin/bash
#SBATCH -J test-biobart-attnpool
#SBATCH -A b16-cbio-ag
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Activate your existing venv (no anaconda/cudnn modules)
source /users/lesego/projects/histgen_dataset/UNI/uni_env/bin/activate

export HF_HOME=/users/lesego/projects/histgen_dataset/.cache/hf
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

# --- Required paths (edit as needed) ---
FEATURES_DIR=/users/lesego/projects/histgen_dataset/test_processed_WSI_features
TEST_JSONL=/users/lesego/projects/histgen_dataset/test.jsonl
CKPT=/users/lesego/projects/histgen_dataset/runs/biobart_attention_lora/best.ckpt
OUT_DIR=/users/lesego/projects/histgen_dataset/runs/biobart_attention_lora_test

mkdir -p "$OUT_DIR" logs

srun python /users/lesego/projects/histgen_dataset/test_biobart_attnpool.py \
  --features_dir "$FEATURES_DIR" \
  --test_jsonl "$TEST_JSONL" \
  --checkpoint "$CKPT" \
  --model_name GanjinZero/biobart-v2-base \
  --batch_size 4 \
  --max_new_tokens 256 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
  --proj_tokens 8 \
  --out_dir "$OUT_DIR"
