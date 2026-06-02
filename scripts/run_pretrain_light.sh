#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PATH="${TRAIN_PATH:-data/nsmc_lm_train.txt}"
VAL_PATH="${VAL_PATH:-data/nsmc_lm_val.txt}"
VOCAB_PATH="${VOCAB_PATH:-data/nsmc_bpe_vocab_2000.json}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/pretrain_light}"
TRAIN_CHAR_LIMIT="${TRAIN_CHAR_LIMIT:-500000}"
VAL_CHAR_LIMIT="${VAL_CHAR_LIMIT:-0}"
VOCAB_SIZE="${VOCAB_SIZE:-2000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-64}"
STRIDE="${STRIDE:-$CONTEXT_LENGTH}"
EMB_DIM="${EMB_DIM:-64}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-1}"
DROP_RATE="${DROP_RATE:-0.0}"
QKV_BIAS="${QKV_BIAS:-false}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
EVAL_FREQ="${EVAL_FREQ:-50}"
EVAL_ITER="${EVAL_ITER:-10}"
CKPT_FREQ="${CKPT_FREQ:-200}"
START_CONTEXT="${START_CONTEXT:-이 영화는}"
SEED="${SEED:-123}"
DEVICE="${DEVICE:-auto}"

if [[ ! -f "$VOCAB_PATH" ]]; then
  echo "vocab 파일이 없습니다: $VOCAB_PATH"
  echo "먼저 bash scripts/build_vocab_light.sh 를 실행하세요."
  exit 1
fi

CMD=(
  "$PYTHON_BIN" run_pretrain.py
  --train-path "$TRAIN_PATH"
  --val-path "$VAL_PATH"
  --vocab-path "$VOCAB_PATH"
  --artifact-dir "$ARTIFACT_DIR"
  --train-char-limit "$TRAIN_CHAR_LIMIT"
  --val-char-limit "$VAL_CHAR_LIMIT"
  --vocab-size "$VOCAB_SIZE"
  --context-length "$CONTEXT_LENGTH"
  --stride "$STRIDE"
  --emb-dim "$EMB_DIM"
  --n-heads "$N_HEADS"
  --n-layers "$N_LAYERS"
  --drop-rate "$DROP_RATE"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --num-epochs "$NUM_EPOCHS"
  --eval-freq "$EVAL_FREQ"
  --eval-iter "$EVAL_ITER"
  --ckpt-freq "$CKPT_FREQ"
  --start-context "$START_CONTEXT"
  --seed "$SEED"
  --device "$DEVICE"
)

if [[ "$QKV_BIAS" == "true" ]]; then
  CMD+=(--qkv-bias)
fi

"${CMD[@]}"
