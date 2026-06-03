# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


import random
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from download_data import _read_nsmc_tsv, _write_jsonl


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    # NSMC 원본 train TSV 파일을 읽어서 [{"text": 리뷰문장, "label": 0 또는 1}, ...] 형태의 리스트로 변환
    train_rows = _read_nsmc_tsv(train_tsv_path)
    # NSMC 원본 test TSV 파일도 같은 형식의 리스트로 변환
    test_rows = _read_nsmc_tsv(test_tsv_path)

    # seed를 고정한 독립적인 랜덤 생성기를 만듦. 같은 seed를 쓰면 매번 같은 방식으로 섞이므로 train/val split이 재현 가능해짐
    rng = random.Random(seed)
    # train 데이터를 무작위로 섞음. 이 뒤에서 앞쪽 일부를 validation으로 떼어낼 것이므로, 원본 순서 편향 없이 train/validation을 나누기 위한 과정
    rng.shuffle(train_rows)

    # train 데이터 전체 개수에 val_ratio를 곱해서 validation 데이터 개수를 계산
    val_size = int(len(train_rows) * val_ratio)
    if len(train_rows) > 0 and val_ratio > 0: # 데이터가 존재하고 validation 비율이 양수라면 최소 1개는 validation으로 둔다.
        val_size = max(1, val_size)
    val_size = min(val_size, len(train_rows)) # validation 샘플 수가 전체 train 샘플 수를 넘지 않도록 제한한다.

    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]
    test_data = test_rows

    # 저장 경로가 주어지면 해당 폴더를 생성한다. 이미 있으면 그대로 사용한다.
    if output_dir is not None: # output_dir가 지정된 경우에만 train/val/test 데이터를 파일로 저장
        output_dir = Path(output_dir) # 문자열 경로가 들어와도 Path 객체로 바꿔서 경로 연산을 쉽게 한다.

        # 저장할 폴더가 없으면 새로 만든다.
        # parents=True: 중간 폴더까지 같이 만든다.
        # exist_ok=True: 이미 폴더가 있어도 에러를 내지 않는다.
        output_dir.mkdir(parents = True, exist_ok = True)

        # output_dir 아래에 train/val/test jsonl 파일을 저장한다.
        # Path 객체에서 / 는 하위 경로를 붙이는 연산자다.
        _write_jsonl(output_dir / "nsmc_sentiment_train.jsonl", train_data)
        _write_jsonl(output_dir / "nsmc_sentiment_val.jsonl", val_data)
        _write_jsonl(output_dir / "nsmc_sentiment_test.jsonl", test_data)

    return train_data, val_data, test_data



class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""

        # Dataset 내부 데이터 리스트에서 idx번째 샘플을 꺼낸다.
        # row 예시: {"text": "이 영화는 정말 좋았다", "label": 1}
        row = self.data[idx]

        # 리뷰 문자열을 BPE tokenizer로 token id 리스트로 변환한다.
        # 감성 분류 = 리뷰 하나 전체를 보고 label 하나 예측 → 문장 경계를 알려주는 BOS/EOS가 있는 편이 자연스러움
        # 예: "재미있다" -> [2, ..., 3]
        token_ids = self.tokenizer.encode(row["text"], add_bos_eos = True) 

        token_ids = token_ids[:self.max_length] # 너무 긴 리뷰는 앞에서 max_length 개만 남김

        # 현재 token 길이가 max_length보다 짧으면 부족한 개수를 계산한다.
        # 예: max_length=8, len(token_ids)=5이면 pad_len=3
        pad_len = self.max_length - len(token_ids)

        # 부족한 길이가 있으면 pad token id를 뒤에 붙여 max_length로 맞춘다.
        # 예: [2, 10, 20, 3] -> [2, 10, 20, 3, 0, 0, 0, 0]
        if pad_len > 0:
            token_ids = token_ids + [self.pad_id] * pad_len
        
        # Python list[int]를 PyTorch tensor로 바꾼다.
        # token id는 embedding lookup에 쓰이므로 dtype은 torch.long이어야 한다.
        input_ids = torch.tensor(token_ids, dtype = torch.long)

        # label을 정수로 변환한다.
        # NSMC label: 0 = 부정, 1 = 긍정
        label = int(row["label"])

        return input_ids, label



class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)
        self.pad_id = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        # token id를 GPT hidden state로 변환한다. (B, T) -> (B, T, emb_dim)
        x = self.gpt.embedding(input_ids)
        x = self.gpt.trf_blocks(x)
        x = self.gpt.final_layernorm(x)

        # PAD가 아닌 실제 토큰 개수를 세고, 마지막 실제 토큰 위치를 구한다.
        lengths = (input_ids != self.pad_id).sum(dim=1).clamp(min=1)
        last_token_idx = lengths - 1

        # 각 배치 샘플에서 마지막 실제 토큰의 hidden state를 문장 대표 벡터로 뽑는다.
        batch_idx = torch.arange(input_ids.size(0), device = input_ids.device)
        pooled = x[batch_idx, last_token_idx]

        # 문장 대표 벡터를 분류 head에 넣어 [부정 점수, 긍정 점수] logits를 만든다.
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        # labels가 있으면 분류 loss까지 계산
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return loss, logits

        return logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    # Dropout 등을 학습 모드로 켠다. 평가 때는 model.eval()을 사용한다.
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # DataLoader는 Dataset.__getitem__이 반환한 (input_ids, label)을 batch 단위로 묶어준다.
    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # PyTorch gradient는 누적되므로, 매 batch마다 이전 gradient를 지운다.
        optimizer.zero_grad()

        # labels를 넘기면 model.forward가 (loss, logits)를 반환한다.
        loss, logits = model(input_ids, labels)

        loss.backward()
        optimizer.step()

        # size(0)은 첫 번째 차원 크기, 즉 현재 batch의 샘플 수다.
        batch_size = input_ids.size(0)

        # cross_entropy의 기본 loss는 batch 평균이므로, 샘플 수를 곱해 합으로 누적한다.
        # 이렇게 해야 마지막 batch 크기가 달라도 epoch 평균 loss를 정확히 계산할 수 있다.
        total_loss += loss.item() * batch_size

        # logits shape: (batch_size, num_labels). dim=1에서 가장 큰 class index를 예측값으로 사용한다.
        # argmax는 가장 큰 값의 index를 반환한다. dim=1은 각 샘플 내부의 class 차원을 뜻한다.
        predictions = torch.argmax(logits, dim = 1)

        # True는 1처럼 더해지므로, 이번 batch에서 맞힌 샘플 수를 계산할 수 있다.
        # item()은 tensor 값 하나를 Python 숫자로 꺼낸다.
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss, logits = model(input_ids, labels)

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size

            predictions = torch.argmax(logits, dim = 1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy