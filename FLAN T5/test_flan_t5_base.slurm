#!/bin/bash
#SBATCH -J test_flan_t5_base
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH --cpus-per-task=4          # define CPUs once
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o test_flan_t5_base.%j.out
#SBATCH -e test_flan_t5_base.%j.err

set -euo pipefail

echo "[INFO] Job $SLURM_JOB_ID on $(hostname) starting at $(date)"

# --- Activate environment ---
source /users/lesego/projects/histgen_dataset/UNI/uni_env/bin/activate

# --- Local cache & tmp (no /scratch) ---
export HF_HOME="/users/lesego/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export NLTK_DATA="/users/lesego/.cache/nltk"
export TMPDIR="/users/lesego/tmp"
mkdir -p "$HF_HOME" "$NLTK_DATA" "$TMPDIR"

# --- Performance environment ---
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# --- Paths ---
FEAT_DIR="/users/lesego/projects/histgen_dataset/test_processed_WSI_features"
TEST_JSONL="/users/lesego/projects/histgen_dataset/test.jsonl"
ADAPTER_DIR="/users/lesego/projects/histgen_dataset/runs/flant5base_pma_k12_len512-seed13"
CKPT="/users/lesego/projects/histgen_dataset/runs/flant5base_pma_k12_len512-seed13/best.ckpt"
OUT_DIR="/users/lesego/projects/histgen_dataset/runs/flant5_eval_${SLURM_JOB_ID}"

# --- Run job ---
echo "[INFO] Launching FLAN-T5 test script ..."
srun python /users/lesego/projects/histgen_dataset/new_test_flan_t5_base_batched.py \
  --feat_dir "$FEAT_DIR" \
  --test_jsonl "$TEST_JSONL" \
  --adapter_dir "$ADAPTER_DIR" \
  ${CKPT:+--ckpt "$CKPT"} \
  --prefix_len 8 \
  --max_new_tokens 256 \
  --num_beams 4 \
  --temperature 1.0 \
  --heartbeat_every 10 \
  --out_dir "$OUT_DIR"

echo "[INFO] Job finished at $(date)"
