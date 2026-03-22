#!/bin/bash
#SBATCH --job-name=golf_smoke
#SBATCH --output=smoke_%j.log
#SBATCH --error=smoke_%j.log
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=00:30:00
#SBATCH --account=jie.xu
#SBATCH --qos=jie.xu

module load conda
conda activate base

cd /blue/jie.xu/mn.hoang/OPEN_AI

# Clone repo if not present
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/openai/parameter-golf.git
fi
cd parameter-golf

# Install deps
pip install --user sentencepiece huggingface-hub datasets tqdm zstandard 2>/dev/null

# Download data (1 shard for smoke test)
if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
fi

# Copy our submission script
cp /blue/jie.xu/mn.hoang/OPEN_AI/2026-03-22_DepthRecurrence/train_gpt.py ./train_gpt_recurrence.py

# Run smoke test: 5 layers x 2 passes, 200 iters
RUN_ID=smoke_depth_recurrence \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=5 \
NUM_RECURRENCE_PASSES=2 \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=262144 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=600 \
SWA_ENABLED=0 \
torchrun --standalone --nproc_per_node=1 ./train_gpt_recurrence.py
