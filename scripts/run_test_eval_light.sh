#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
TEST_PATH="${TEST_PATH:-data/nsmc_sentiment_test.jsonl}"
VOCAB_PATH="${VOCAB_PATH:-data/nsmc_bpe_vocab_2000.json}"
FINETUNED_CHECKPOINT="${FINETUNED_CHECKPOINT:-artifacts/finetune_light/checkpoints/best.pt}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/test_eval_light}"
VOCAB_SIZE="${VOCAB_SIZE:-2000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-64}"
EMB_DIM="${EMB_DIM:-64}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-1}"
DROP_RATE="${DROP_RATE:-0.0}"
QKV_BIAS="${QKV_BIAS:-false}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-auto}"

if [[ ! -f "$VOCAB_PATH" ]]; then
  echo "vocab 파일이 없습니다: $VOCAB_PATH"
  echo "먼저 bash scripts/build_vocab_light.sh 를 실행하세요."
  exit 1
fi

if [[ ! -f "$FINETUNED_CHECKPOINT" ]]; then
  echo "finetune checkpoint가 없습니다: $FINETUNED_CHECKPOINT"
  echo "먼저 bash scripts/run_finetune_light.sh 를 실행하세요."
  exit 1
fi

CMD=(
  "$PYTHON_BIN" run_test_eval.py
  --test-path "$TEST_PATH"
  --vocab-path "$VOCAB_PATH"
  --finetuned-checkpoint "$FINETUNED_CHECKPOINT"
  --artifact-dir "$ARTIFACT_DIR"
  --vocab-size "$VOCAB_SIZE"
  --context-length "$CONTEXT_LENGTH"
  --emb-dim "$EMB_DIM"
  --n-heads "$N_HEADS"
  --n-layers "$N_LAYERS"
  --drop-rate "$DROP_RATE"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
)

if [[ "$QKV_BIAS" == "true" ]]; then
  CMD+=(--qkv-bias)
fi

"${CMD[@]}"
