# W14-gpt-lab 실험 실행 가이드

## 1. 이 문서의 역할

이 문서는 **현재 저장소에 이미 들어간 실험 실행 구조**를 설명합니다.

- Colab에서는 `gpt-lab.ipynb` 맨 뒤의 `실험 설정 셀` 1개, 실행 셀 5개, 요약 셀 2개를 사용합니다.
- 로컬/터미널에서는 `scripts/*.sh`를 사용합니다.
- 실제 실행은 shell wrapper -> `run_*.py` runner -> `src/*` 모듈 순서로 이어집니다.

실험 해석 원칙 자체는 [docs/howToExperiment.md](/Users/wiseungcheol/Desktop/LLM_project/docs/howToExperiment.md)를 따릅니다.  
이 문서는 **baseline에서 시작해 곡선을 보고 한 번에 하나씩 바꾸는 방법**과 **어디를 바꾸면 실제로 무엇이 바뀌는지**에 집중합니다.

## 2. 전체 흐름

권장 실험 흐름은 아래 5단계입니다.

1. `prepare_nsmc.sh`
2. `build_vocab_light.sh`
3. `run_pretrain_light.sh`
4. `run_finetune_light.sh`
5. `run_test_eval_light.sh`

Colab에서는 노트북 맨 뒤 실행 셀 5개가 이 순서를 그대로 실행합니다.  
로컬에서는 아래처럼 실행하면 됩니다.

```bash
cd W14-gpt-lab

PYTHON_BIN=../.venv/bin/python bash scripts/prepare_nsmc.sh
PYTHON_BIN=../.venv/bin/python bash scripts/build_vocab_light.sh
PYTHON_BIN=../.venv/bin/python bash scripts/run_pretrain_light.sh
PYTHON_BIN=../.venv/bin/python bash scripts/run_finetune_light.sh
PYTHON_BIN=../.venv/bin/python bash scripts/run_test_eval_light.sh
```

## 3. 어디에 뭐가 있나

실행 엔트리:

- [run_build_vocab.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/run_build_vocab.py)
- [run_pretrain.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/run_pretrain.py)
- [run_finetune.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/run_finetune.py)
- [run_test_eval.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/run_test_eval.py)

셸 래퍼:

- [scripts/prepare_nsmc.sh](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/scripts/prepare_nsmc.sh)
- [scripts/build_vocab_light.sh](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/scripts/build_vocab_light.sh)
- [scripts/run_pretrain_light.sh](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/scripts/run_pretrain_light.sh)
- [scripts/run_finetune_light.sh](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/scripts/run_finetune_light.sh)
- [scripts/run_test_eval_light.sh](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/scripts/run_test_eval_light.sh)

노트북:

- [gpt-lab.ipynb](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/gpt-lab.ipynb)

## 4. Colab에서 어떻게 쓰나

노트북 맨 뒤에는 아래 순서로 셀이 있습니다.

1. `실험 설정 셀`
2. `prepare` 실행 셀
3. `vocab` 실행 셀
4. `pretrain` 실행 셀
5. `finetune` 실행 셀
6. `test 평가` 실행 셀
7. `실험 로그 요약 셀`
8. `누적 실험 표 보기 셀`

반복 실험할 때는 **실험 설정 셀만 바꾸고 필요한 실행 셀만 다시 실행**하면 됩니다.
다만 발표용 비교를 위해서는 **baseline 대비 바꾼 항목을 run마다 하나만 유지**하는 것을 기본 원칙으로 잡습니다.

현재 실험 설정 셀 기본값은 아래와 같습니다.

### 4.1 공통 / 실험 관리

- `RUN_NAME = "light_baseline"`
- `STAGE = "pretrain"`
- `CHANGED_HYPERPARAM = ""`
- `CHANGED_VALUE = ""`
- `RUN_NOTE = ""`
- `NEXT_ACTION = ""`
- `PYTHON_BIN = "python"`
- `SEED = 123`
- `DEVICE = "auto"`

이 값들은 아래 경로 계산에도 사용됩니다.

- `VOCAB_PATH = f"data/nsmc_bpe_vocab_{VOCAB_SIZE}.json"`
- `PRETRAIN_ARTIFACT_DIR = f"artifacts/{RUN_NAME}/pretrain"`
- `FINETUNE_ARTIFACT_DIR = f"artifacts/{RUN_NAME}/finetune"`
- `PRETRAINED_CHECKPOINT = f"{PRETRAIN_ARTIFACT_DIR}/checkpoints/best.pt"`
- `FINETUNED_CHECKPOINT = f"{FINETUNE_ARTIFACT_DIR}/checkpoints/best.pt"`
- `TEST_EVAL_ARTIFACT_DIR = f"artifacts/{RUN_NAME}/test_eval"`
- `EXPERIMENT_LOG_PATH = "artifacts/experiment_log.csv"`

### 4.2 vocab / pretrain 설정

- `TRAIN_CHAR_LIMIT = 500000`
- `VAL_CHAR_LIMIT = 0`
- `TOKENIZER_CORPUS_CHARS = TRAIN_CHAR_LIMIT`
- `VAL_SPLIT_RATIO = 0.08`
- `VOCAB_SIZE = 2000`
- `CONTEXT_LENGTH = 64`
- `STRIDE = CONTEXT_LENGTH`
- `EMB_DIM = 64`
- `N_HEADS = 4`
- `N_LAYERS = 1`
- `DROP_RATE = 0.0`
- `QKV_BIAS = False`
- `BATCH_SIZE = 8`
- `LEARNING_RATE = "3e-4"`
- `WEIGHT_DECAY = "0.0"`
- `NUM_EPOCHS = 5`
- `EVAL_FREQ = 50`
- `EVAL_ITER = 10`
- `CKPT_FREQ = 200`
- `START_CONTEXT = "이 영화는"`

### 4.3 finetune 설정

- `FINETUNE_BATCH_SIZE = 16`
- `FINETUNE_LEARNING_RATE = "5e-5"`
- `FINETUNE_WEIGHT_DECAY = "0.0"`
- `FINETUNE_NUM_EPOCHS = 2`
- `FINETUNE_DROP_RATE = 0.1`

### 4.4 test 평가 설정

- `TEST_EVAL_ARTIFACT_DIR`
- `FINETUNED_CHECKPOINT`

### 4.5 발표용 로그 설정

- `STAGE`
- `CHANGED_HYPERPARAM`
- `CHANGED_VALUE`
- `RUN_NOTE`
- `NEXT_ACTION`

## 5. 설정 값이 실제 어디에 반영되나

현재 구조에서 값 전달은 아래처럼 됩니다.

### 5.1 pretrain

노트북 설정 셀
-> `scripts/run_pretrain_light.sh`
-> `run_pretrain.py`
-> `GPTModel(config)` / `create_dataloader(...)` / `train_model(...)`

실제로 반영되는 항목:

- `VOCAB_SIZE`
  - `BPETokenizer(vocab_size=...)`
  - `GPTModel(config["vocab_size"])`
- `CONTEXT_LENGTH`
  - `create_dataloader(..., context_length=...)`
  - `GPTModel(config["context_length"])`
- `STRIDE`
  - `create_dataloader(..., stride=...)`
  - 현재 baseline에서는 `STRIDE = CONTEXT_LENGTH`
- `EMB_DIM`, `N_HEADS`, `N_LAYERS`, `DROP_RATE`, `QKV_BIAS`
  - `GPTModel(config)`에 그대로 반영
- `BATCH_SIZE`
  - train/val dataloader 둘 다에 반영
- `LEARNING_RATE`, `WEIGHT_DECAY`
  - `torch.optim.AdamW(...)`에 반영
- `NUM_EPOCHS`, `EVAL_FREQ`, `EVAL_ITER`, `CKPT_FREQ`, `START_CONTEXT`
  - `train_model(...)` 호출 인자에 반영

### 5.2 finetune

노트북 설정 셀
-> `scripts/run_finetune_light.sh`
-> `run_finetune.py`
-> `ReviewSentimentDataset(...)` / `GPTModel(config)` / `GPTForSequenceClassification(...)`

실제로 반영되는 항목:

- `VOCAB_SIZE`
  - tokenizer load와 backbone config에 반영
- `CONTEXT_LENGTH`
  - `ReviewSentimentDataset(..., max_length=...)`
  - backbone `config["context_length"]`
- `EMB_DIM`, `N_HEADS`, `N_LAYERS`, `QKV_BIAS`
  - backbone `config`에 반영
- `FINETUNE_DROP_RATE`
  - backbone config의 `drop_rate`
  - `GPTForSequenceClassification(..., drop_rate=...)`
- `FINETUNE_BATCH_SIZE`
  - train/val DataLoader에 반영
- `FINETUNE_LEARNING_RATE`, `FINETUNE_WEIGHT_DECAY`
  - classifier optimizer에 반영
- `FINETUNE_NUM_EPOCHS`
  - epoch loop 횟수에 반영
- `PRETRAINED_CHECKPOINT`
  - backbone 가중치 load 경로에 반영

### 5.3 test-only 평가

노트북 설정 셀
-> `scripts/run_test_eval_light.sh`
-> `run_test_eval.py`
-> `ReviewSentimentDataset(...)` / `GPTForSequenceClassification(...)` / `evaluate_sentiment(...)`

실제로 반영되는 항목:

- `VOCAB_SIZE`, `CONTEXT_LENGTH`, `EMB_DIM`, `N_HEADS`, `N_LAYERS`, `QKV_BIAS`
  - test 평가용 모델 구조에 반영
- `FINETUNE_DROP_RATE`
  - 평가용 분류 모델 생성 시 사용
- `FINETUNE_BATCH_SIZE`
  - test DataLoader batch size에 반영
- `FINETUNED_CHECKPOINT`
  - fine-tuned classifier checkpoint load 경로에 반영

### 5.4 발표용 로그 요약

노트북 설정 셀
-> `experiment_log.py`
-> `artifacts/experiment_log.csv`

기록되는 핵심 항목:

- `run`, `stage`
- `changed_hyperparam`, `changed_value`
- `train_loss`, `val_loss`
- `note`, `next_action`
- pretrain: `best_val_loss`, `last_train_loss`, `val_ppl`, `train_elapsed_seconds`
- finetune: `best_val_loss`, `best_val_acc`, `last_train_loss`, `fit_elapsed_seconds`

## 6. 데이터 / 산출물

### 6.1 prepare 이후

생성 파일:

- `data/nsmc_lm_train.txt`
- `data/nsmc_lm_val.txt`
- `data/nsmc_sentiment_train.jsonl`
- `data/nsmc_sentiment_val.jsonl`
- `data/nsmc_sentiment_test.jsonl`

### 6.2 vocab 이후

생성 파일:

- `data/nsmc_bpe_vocab_2000.json`

주의:

- `VOCAB_SIZE`나 tokenizer corpus를 바꾼 run은 token-level `val loss`를 1:1로 직접 비교하면 해석이 흔들릴 수 있습니다.
- 이런 실험은 `val loss`와 함께 시퀀스 길이 변화, 학습 시간, downstream 성능도 같이 기록하는 편이 안전합니다.

### 6.3 pretrain 이후

기본 출력 경로:

- `artifacts/<RUN_NAME>/pretrain/`

생성 파일:

- `metrics.jsonl`
- `loss.png`
- `samples.txt`
- `timing.json`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `checkpoints/checkpoint_step_*.pt`

### 6.4 finetune 이후

기본 출력 경로:

- `artifacts/<RUN_NAME>/finetune/`

생성 파일:

- `metrics.jsonl`
- `loss.png`
- `accuracy.png`
- `timing.json`
- `checkpoints/best.pt`
- `checkpoints/last.pt`

### 6.5 test 평가 이후

기본 출력 경로:

- `artifacts/<RUN_NAME>/test_eval/`

생성 파일:

- `test_metrics.json`
- `timing.json`

### 6.6 발표용 로그 이후

기본 출력 경로:

- `artifacts/experiment_log.csv`

이 파일은 노트북의 `실험 로그 요약 셀`이 자동으로 overwrite/update 합니다.  
같은 `run + stage` 조합은 덮어쓰고, 새로운 실험 조합은 새 줄로 추가합니다.

## 7. 모델 저장 / 불러오기

### 7.1 pretrain checkpoint

저장 함수:

- [src/train.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/src/train.py:60) 의 `save_checkpoint()`

저장 항목:

- `model_state_dict`
- `optimizer_state_dict`
- `epoch`
- `global_step`

불러오기 함수:

- [src/train.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/src/train.py:77) 의 `load_checkpoint()`

사용 예:

```python
from src.model import GPTModel
from src.train import load_checkpoint
import torch

config = {
    "vocab_size": 2000,
    "context_length": 64,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}

model = GPTModel(config)
epoch, step = load_checkpoint(
    model,
    optimizer=None,
    path="artifacts/light_baseline/pretrain/checkpoints/best.pt",
    device=torch.device("cpu"),
)
```

### 7.2 finetune checkpoint

저장 함수:

- [run_finetune.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/run_finetune.py:120) 의 `save_classifier_checkpoint()`

저장 항목:

- `model_state_dict`
- `optimizer_state_dict`
- `epoch`

현재는 fine-tuning 결과를 저장하지만, **fine-tuning resume 기능은 아직 없습니다.**

## 8. 실험 중 바꾸기 쉬운 값

현재 노트북/셸에서 바로 바꿀 수 있는 값은 아래입니다.

### 8.1 pretrain 쪽

- `RUN_NAME`
- `STAGE`
- `CHANGED_HYPERPARAM`
- `CHANGED_VALUE`
- `RUN_NOTE`
- `NEXT_ACTION`
- `TRAIN_CHAR_LIMIT`
- `VAL_CHAR_LIMIT`
- `TOKENIZER_CORPUS_CHARS`
- `VOCAB_SIZE`
- `CONTEXT_LENGTH`
- `STRIDE`
- `EMB_DIM`
- `N_HEADS`
- `N_LAYERS`
- `DROP_RATE`
- `QKV_BIAS`
- `BATCH_SIZE`
- `LEARNING_RATE`
- `WEIGHT_DECAY`
- `NUM_EPOCHS`
- `EVAL_FREQ`
- `EVAL_ITER`
- `CKPT_FREQ`
- `START_CONTEXT`
- `SEED`
- `DEVICE`

### 8.2 finetune 쪽

- `RUN_NAME`
- `CONTEXT_LENGTH`
- `EMB_DIM`
- `N_HEADS`
- `N_LAYERS`
- `QKV_BIAS`
- `FINETUNE_BATCH_SIZE`
- `FINETUNE_LEARNING_RATE`
- `FINETUNE_WEIGHT_DECAY`
- `FINETUNE_NUM_EPOCHS`
- `FINETUNE_DROP_RATE`
- `PRETRAINED_CHECKPOINT`
- `SEED`
- `DEVICE`

발표용 실험 기록도 현재 노트북 셀에서 직접 바꿀 수 있습니다.

- `STAGE`
- `CHANGED_HYPERPARAM`
- `CHANGED_VALUE`
- `RUN_NOTE`
- `NEXT_ACTION`

## 9. 지금 테스트 환경에서 자동으로 확인되는 것

현재 아래 항목은 자동 테스트가 있습니다.

- `tests/test_train.py`
  - `plot_losses(..., output_path=...)`
  - `train_model()` 반환값과 checkpoint 생성
- `tests/test_finetune.py`
  - dataset / classifier / train-eval 함수 기본 동작
- `tests/test_runners.py`
  - `run_build_vocab.py`, `run_pretrain.py`, `run_finetune.py`, `run_test_eval.py` 의 `--help`
  - 셸 5개 실행 권한
  - pretrain runner 인자가 실제 model/dataloader/train loop로 전달되는지
  - finetune runner 인자가 실제 train/val dataset, model, classifier, epoch loop로 전달되는지
  - test eval runner 인자가 실제 test dataset, model, evaluate loop로 전달되는지
- `tests/test_experiment_log.py`
  - pretrain/finetune 요약 로직이 `metrics.jsonl`, `timing.json`에서 올바른 값을 집계하는지
  - `experiment_log.csv`가 같은 `run + stage` 조합에서 overwrite 되는지
  - 노트북 10장에 요약 셀과 vocab 경고 문구가 있는지

즉, 지금은 **설정 셀에서 바꾼 핵심 실험 값이 실제 runner 내부에서 먹는지**는 자동으로 확인되는 상태입니다.

## 10. 아직 자동화되지 않은 것

아래는 아직 사람이 직접 확인해야 합니다.

- 실제 `500000`자 vocab 빌드 시간
- full pretrain end-to-end 시간과 메모리
- full finetune end-to-end 시간과 메모리
- Colab GPU에서의 장시간 안정성
- pretrain / finetune resume

그래도 반복 실험 자체는 지금 꽤 편한 편입니다.

- 노트북에서 설정 셀만 바꾸면 됩니다.
- `RUN_NAME`만 바꿔도 artifact 경로가 분리됩니다.
- `PRETRAINED_CHECKPOINT`도 기본적으로 `RUN_NAME` 기준 pretrain 결과를 가리킵니다.
