#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PATH="${TRAIN_PATH:-data/nsmc_sentiment_train.jsonl}"
VAL_PATH="${VAL_PATH:-data/nsmc_sentiment_val.jsonl}"
VOCAB_PATH="${VOCAB_PATH:-data/nsmc_bpe_vocab_2000.json}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-artifacts/pretrain_light/checkpoints/best.pt}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/finetune_light}"
VOCAB_SIZE="${VOCAB_SIZE:-2000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-64}"
EMB_DIM="${EMB_DIM:-64}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-1}"
DROP_RATE="${DROP_RATE:-0.1}"
QKV_BIAS="${QKV_BIAS:-false}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
EVAL_FREQ="${EVAL_FREQ:-1500}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-123}"
DEVICE="${DEVICE:-auto}"

if [[ ! -f "$VOCAB_PATH" ]]; then
  echo "vocab 파일이 없습니다: $VOCAB_PATH"
  echo "먼저 bash scripts/build_vocab_light.sh 를 실행하세요."
  exit 1
fi

if [[ -n "$PRETRAINED_CHECKPOINT" && ! -f "$PRETRAINED_CHECKPOINT" ]]; then
  echo "pretrain checkpoint가 없습니다: $PRETRAINED_CHECKPOINT"
  echo "먼저 bash scripts/run_pretrain_light.sh 를 실행하세요."
  exit 1
fi

CMD=(
  "$PYTHON_BIN" run_finetune.py
  --train-path "$TRAIN_PATH"
  --val-path "$VAL_PATH"
  --vocab-path "$VOCAB_PATH"
  --pretrained-checkpoint "$PRETRAINED_CHECKPOINT"
  --artifact-dir "$ARTIFACT_DIR"
  --vocab-size "$VOCAB_SIZE"
  --context-length "$CONTEXT_LENGTH"
  --emb-dim "$EMB_DIM"
  --n-heads "$N_HEADS"
  --n-layers "$N_LAYERS"
  --drop-rate "$DROP_RATE"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --num-epochs "$NUM_EPOCHS"
  --eval-freq "$EVAL_FREQ"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --device "$DEVICE"
)

if [[ "$QKV_BIAS" == "true" ]]; then
  CMD+=(--qkv-bias)
fi

"${CMD[@]}"
