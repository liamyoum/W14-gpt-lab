# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

import csv
import json
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def _clean_text(text: str | None) -> str:
    """TSV에서 읽은 리뷰 텍스트를 한 줄 문자열로 정리합니다."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _read_sentiment_tsv(path: str | Path) -> list[dict]:
    """NSMC 형식 TSV를 읽어 {text, label} 리스트로 변환합니다."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = _clean_text(row.get("document"))
            label = row.get("label")
            if not text or label not in {"0", "1"}:
                continue
            rows.append({"text": text, "label": int(label)})
    return rows


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """분류 데이터를 JSONL 한 줄 한 샘플 형식으로 저장합니다."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    # 원본 train TSV에서 감성 분류용 샘플을 읽습니다.
    train_rows = _read_sentiment_tsv(train_tsv_path)

    # 재현 가능하게 validation split을 만들기 위해 고정 seed로 섞습니다.
    rng = random.Random(seed)
    rng.shuffle(train_rows)

    # validation 크기를 먼저 정하고 나머지를 train으로 사용합니다.
    val_size = max(1, int(len(train_rows) * val_ratio)) if train_rows else 0
    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]

    # 별도 test TSV가 있으면 읽고, 없으면 빈 test set을 반환합니다.
    test_data = _read_sentiment_tsv(test_tsv_path) if test_tsv_path is not None else []

    # 요청 시 현재 split 결과를 재사용 가능한 JSONL 파일로도 남깁니다.
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
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
        # 현재 샘플 하나만 꺼내서 text와 label을 읽는다.
        sample = self.data[idx]
        encoded = self.tokenizer.encode(sample["text"], add_bos_eos=True)

        # max_length를 기준으로 너무 긴 시퀀스는 자르고, 부족한 길이는 pad_id로 오른쪽을 채운다.
        encoded = encoded[: self.max_length]
        if len(encoded) < self.max_length:
            encoded = encoded + [self.pad_id] * (self.max_length - len(encoded))

        return torch.tensor(encoded, dtype=torch.long), sample["label"]


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
        # GPT backbone은 고정하고 새로 추가한 분류층만 학습합니다.
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)
        for param in self.gpt.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.
        
        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        # LM head 대신 backbone hidden state를 바로 분류에 사용합니다.
        x = self.gpt.tok_emb(input_ids)
        x = x + self.gpt.pos_emb(torch.arange(input_ids.size(1), device=input_ids.device))
        x = self.gpt.drop_emb(x)
        x = self.gpt.trf_blocks(x)
        x = self.gpt.final_norm(x)

        # padding을 제외한 마지막 토큰 hidden state를 문장 표현으로 사용합니다.
        last_idx = input_ids.ne(0).sum(dim=1).sub(1).clamp(min=0)
        pooled = x[torch.arange(x.size(0), device=x.device), last_idx]
        logits = self.classifier(self.dropout(pooled))

        if labels is None:
            return logits

        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, labels in train_loader:
        # 배치 데이터를 현재 device로 옮깁니다.
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # 분류 loss를 계산하고 classifier 파라미터를 업데이트합니다.
        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        # epoch 평균을 위해 loss와 accuracy 통계를 누적합니다.
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.size(0)
    # epoch 마다 loss, accur 집계
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_count
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
    total_count = 0

    # 평가는 gradient 없이 같은 지표만 계산합니다.
    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss, logits = model(input_ids, labels=labels)
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_count
    return avg_loss, accuracy
