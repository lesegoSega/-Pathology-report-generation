#!/bin/bash
#SBATCH -J bart-attnpool
#SBATCH -A b16-cbio-ag
#SBATCH -p JupyterGPU
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# #SBATCH --mail-user=you@email
# #SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge || true
# module load CUDA/12.1

source /users/lesego/projects/histgen_dataset/UNI/uni_env/bin/activate

export HF_HOME=/users/lesego/.cache/huggingface
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

TMPBASE="${SLURM_TMPDIR:-/users/lesego/tmp}"
mkdir -p "$TMPBASE"
TMPDIR="$(mktemp -d "$TMPBASE/bart_attnpool.${SLURM_JOB_ID:-local}.XXXXXX")"
export TMPDIR
echo "[INFO] Using TMPDIR=$TMPDIR"
cleanup() {
  if [[ "$TMPDIR" == /users/lesego/tmp/* ]]; then
    rm -rf "$TMPDIR" || true
  fi
}
trap cleanup EXIT

mkdir -p logs

FEAT_DIR=/users/lesego/projects/histgen_dataset/processed_WSI_features
REPORTS=/users/lesego/projects/histgen_dataset/reports.jsonl
SPLITS=/users/lesego/projects/histgen_dataset/splits.json
ORGAN=/users/lesego/projects/histgen_dataset/slide_to_organ.csv
LEXICON=/users/lesego/projects/histgen_dataset/organ_terms.csv

OUTDIR=/users/lesego/projects/histgen_dataset/runs/bart_base_attnpool_lora

srun python /users/lesego/projects/histgen_dataset/train_bart_attnpool.py \
  --feat_dir "$FEAT_DIR" \
  --reports_jsonl "$REPORTS" \
  --splits "$SPLITS" \
  --organ_csv "$ORGAN" \
  --lexicon_csv "$LEXICON" \
  --text_model_name facebook/bart-base \
  --k_prefix 24 \
  --n_heads 8 \
  --n_layers 2 \
  --dropout 0.15 \
  --max_patches 2048 \
  --max_source_len 1024 \
  --label_max_len 768 \
  --max_target_len 768 \
  --min_decode_len 300 \
  --no_repeat_ngram_size 4 \
  --num_beams 4 \
  --length_penalty 1.15 \
  --epochs 12 \
  --train_bsz 1 \
  --eval_bsz 1 \
  --lr 7e-5 \
  --warmup_steps 800 \
  --weight_decay 0.02 \
  --grad_accum 16 \
  --max_grad_norm 1.0 \
  --label_smoothing 0.10 \
  --lora_r 32 \
  --lora_alpha 16 \
  --lora_dropout 0.15 \
  --grad_checkpoint \
  --out_dir "$OUTDIR"
