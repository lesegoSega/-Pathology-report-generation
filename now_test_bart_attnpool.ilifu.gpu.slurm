#!/bin/bash
#SBATCH -J test_bart_attnpool       # Job name
#SBATCH -A b16-cbio-ag              # Your project/account name
#SBATCH -p GPU                      # Batch partition for GPU
#SBATCH --gres=gpu:1                # Requesting 1 GPU
#SBATCH -c 4                        # 4 CPU cores
#SBATCH --mem=16G                   # Memory allocation
#SBATCH --time=2-00:00:00           # Maximum wall time (2 days)
#SBATCH -o logs/test_bart-%j.out    # Standard output log
#SBATCH -e logs/test_bart-%j.err    # Standard error log
#SBATCH --mail-user=your_email@example.com  # Email notification (optional)
#SBATCH --mail-type=BEGIN,END,FAIL # Email notification types

# Load necessary modules
module load cuda/11.7              # CUDA version, adjust as needed
module load anaconda/2021.05       # Adjust Anaconda module version
module load python/3.8.5           # Load Python version, adjust as needed

# Activate the environment
source /users/lesego/projects/histgen_dataset/UNI/uni_env/bin/activate

# Run the testing script
python /users/lesego/projects/histgen_dataset/now_test_bart_attnpool.py \
  --feat_dir /users/lesego/projects/histgen_dataset/test_features \
  --test_jsonl /users/lesego/projects/histgen_dataset/test.jsonl \
  --ckpt_dir /users/lesego/projects/histgen_dataset/runs/bart_base_attnpool_lora \
  --hf_backbone facebook/bart-base \
  --batch_size 2 \
  --max_new_tokens 256 \
  --seed 13 \
  --out_dir /users/lesego/projects/histgen_dataset/runs/bart_present_only_eval
