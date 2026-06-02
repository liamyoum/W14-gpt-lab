#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PATH="${TRAIN_PATH:-data/nsmc_lm_train.txt}"
TRAIN_CHAR_LIMIT="${TRAIN_CHAR_LIMIT:-500000}"
VOCAB_SIZE="${VOCAB_SIZE:-2000}"
OUTPUT_PATH="${OUTPUT_PATH:-data/nsmc_bpe_vocab_2000.json}"

"$PYTHON_BIN" run_build_vocab.py \
  --train-path "$TRAIN_PATH" \
  --train-char-limit "$TRAIN_CHAR_LIMIT" \
  --vocab-size "$VOCAB_SIZE" \
  --output-path "$OUTPUT_PATH"
