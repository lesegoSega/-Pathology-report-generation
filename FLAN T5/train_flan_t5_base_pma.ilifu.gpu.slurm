#!/bin/bash
#SBATCH -J flanT5b-PMA-K12
#SBATCH -A b16-cbio-ag
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# #SBATCH --mail-user=you@email
# #SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
mkdir -p logs runs

source /users/lesego/projects/histgen_dataset/UNI/uni_env/bin/activate
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
python - <<'PY'
import torch; torch.set_float32_matmul_precision("high")
print("matmul precision set to 'high'")
PY

python /users/lesego/projects/histgen_dataset/train_t5_pma.py \
  --model_name google/flan-t5-base \
  --feat_dir /users/lesego/projects/histgen_dataset/processed_WSI_features \
  --reports  /users/lesego/projects/histgen_dataset/reports.jsonl \
  --splits   /users/lesego/projects/histgen_dataset/splits.json \
  --pma_k 12 --pma_heads 8 --pma_dropout 0.1 \
  --max_patches 4096 --subsample_mode uniform \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --lr 5e-4 --weight_decay 0.0 --dropout 0.1 \
  --batch_size 2 --grad_accum 16 \
  --epochs 35 \
  --max_input_len 48 --max_target_len 512 \
  --min_new_tokens 160 --max_new_tokens 512 \
  --num_beams 6 --length_penalty 1.2 \
  --hint_source report \
  --use_amp --amp_dtype bf16 \
  --exp_name flant5base_pma_k12_len512 --seed 13
