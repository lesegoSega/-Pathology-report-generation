#!/bin/bash
#SBATCH -J biobart-attnpool-lora
#SBATCH -A b16-cbio-ag
#SBATCH -p GPU                  # ilifu GPU partition name from your working example
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# #SBATCH --mail-user=you@email
# #SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# --- Environment ---
module purge || true
module load cuda/12.4 || true        # adjust if your CUDA module differs
# DO NOT load 'nccl' (it doesn't exist as a separate module on ilifu)
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/users/$USER/projects/histgen_dataset/.cache/hf"
mkdir -p "$HF_HOME" logs

source /users/$USER/projects/histgen_dataset/UNI/uni_env/bin/activate

# --- Paths (edit if your paths differ) ---
FEATS="/users/$USER/projects/histgen_dataset/processed_WSI_features"
REPORTS="/users/$USER/projects/histgen_dataset/reports.jsonl"
SPLITS="/users/$USER/projects/histgen_dataset/splits.json"
RUNDIR="/users/$USER/projects/histgen_dataset/runs/biobart_attention_lora"

# --- Training ---
srun python -u /users/$USER/projects/histgen_dataset/train_biobart.py \
  --feat_dir "$FEATS" \
  --reports_jsonl "$REPORTS" \
  --splits "$SPLITS" \
  --run_dir "$RUNDIR" \
  --model_name "GanjinZero/biobart-v2-base" \
  --epochs 10 \
  --bsz 4 \
  --lr 3e-5 \
  --warmup 500 \
  --num_workers 0 \
  --k_resampler 12 \
  --max_input_tokens 512 \
  --max_target_tokens 640 \
  --use_lora \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05
