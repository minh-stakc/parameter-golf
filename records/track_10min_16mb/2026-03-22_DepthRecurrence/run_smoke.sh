#!/bin/bash
# Quick smoke test on 1xA100 (~3 min, 200 iters)
# Usage: bash records/track_10min_16mb/2026-03-22_DepthRecurrence/run_smoke.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../../.."

# Download data if not present
if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
    echo "Downloading data (1 shard for smoke test)..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
fi

RUN_ID=smoke_depth_recurrence \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=5 \
NUM_RECURRENCE_PASSES=2 \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=262144 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=180 \
SWA_ENABLED=0 \
torchrun --standalone --nproc_per_node=1 \
    "$SCRIPT_DIR/train_gpt.py"
