#!/bin/bash
#SBATCH -J uni-wsi-gpu
#SBATCH -A b16-cbio-ag
#SBATCH -p GPU
#SBATCH --gres=gpu:1              # set to 2 if you want two GPUs
#SBATCH -c 16
#SBATCH --mem=220G
#SBATCH --time=5-00:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# #SBATCH --mail-user=you@email   # (optional) uncomment + set
# #SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ---- Paths (edit if needed) ----
PROJECT_ROOT="/users/lesego/projects/histgen_dataset"
UNI_ROOT="${PROJECT_ROOT}/UNI"
MANIFEST="${PROJECT_ROOT}/UNI/gdc_manifest_part_001.txt"
OUT_DIR="${PROJECT_ROOT}/processed_WSI_features"
GDC_CLIENT="${PROJECT_ROOT}/gdc-client"
CACHE_DIR="${PROJECT_ROOT}/.hf_cache"

mkdir -p "${OUT_DIR}" "${CACHE_DIR}" logs

# Optional: system libs
module load openslide || true

# ENV
source "${UNI_ROOT}/uni_env/bin/activate"
export HF_HOME="${CACHE_DIR}"
export TRANSFORMERS_CACHE="${CACHE_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "Node: $(hostname)"
nvidia-smi || true

# Work in code dir so relative paths work
cd "${UNI_ROOT}"

# ---- GPU settings ----
# If you request --gres=gpu:2, set GPU_IDS="0,1" and NUM_WORKERS=2
GPU_IDS="0"
NUM_WORKERS=1
PATCH=224
STEP=224
BATCH=32
WHITE=0.8
CKPT=5
TMP_ROOT="${SLURM_TMPDIR:-/tmp}"

# Preflight (quick)
python - <<'PY'
import os, sys
print("CWD:", os.getcwd())
print("Python:", sys.version)
print("Pipeline exists:", os.path.exists("multi_wsi_uni_pipeline_part1.py"))
PY

# ---- Run ----
srun -u python multi_wsi_uni_pipeline_part1.py \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --gdc-client "${GDC_CLIENT}" \
  --num-workers "${NUM_WORKERS}" \
  --gpu-ids "${GPU_IDS}" \
  --patch-size "${PATCH}" \
  --step-size "${STEP}" \
  --batch-size "${BATCH}" \
  --white-thresh "${WHITE}" \
  --checkpoint-every "${CKPT}" \
  --tmp-root "${TMP_ROOT}"
