#!/bin/bash
# Full training run on 1xA100 (~80 min, equivalent to 10min on 8xH100)
# Usage: SEED=42 bash records/track_10min_16mb/2026-03-22_DepthRecurrence/run_full.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../../.."

# Download full data if not present
if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
    echo "Downloading data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

SEED="${SEED:-42}"

RUN_ID="depth_recurrence_seed${SEED}" \
SEED="$SEED" \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=5 \
NUM_RECURRENCE_PASSES=2 \
MAX_WALLCLOCK_SECONDS=4800 \
torchrun --standalone --nproc_per_node=1 \
    "$SCRIPT_DIR/train_gpt.py"
