# -*- coding: utf-8 -*-
"""NSMC Light baseline용 test-only 감성 분류 평가 러너."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe import BPETokenizer
from src.finetune import GPTForSequenceClassification, ReviewSentimentDataset, evaluate_sentiment
from src.model import GPTModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test-only evaluation for NSMC sentiment classifier.")
    parser.add_argument("--test-path", type=Path, required=True, help="Sentiment test JSONL path.")
    parser.add_argument("--vocab-path", type=Path, required=True, help="Saved BPE vocab JSON path.")
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True, help="Fine-tuned classifier checkpoint path.")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Directory for test metrics output.")
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--qkv-bias", action="store_true", help="Enable QKV bias.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, mps")
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"sentiment JSONL 파일이 없습니다: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"비어 있는 JSONL 입니다: {path}")
    return rows


def write_timing_json(path: Path, timing: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timing, f, ensure_ascii=False, indent=2)


def load_finetuned_checkpoint(
    model: GPTForSequenceClassification,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """동적 causal mask buffer 차이를 무시하고 분류기 checkpoint를 복원합니다."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if not key.endswith(".att.mask")
    }
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    missing_keys = [key for key in missing_keys if not key.endswith(".att.mask")]
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "finetune checkpoint와 현재 분류기 구조가 맞지 않습니다. "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )


def main() -> None:
    args = parse_args()
    if not args.vocab_path.exists():
        raise SystemExit(
            f"vocab 파일이 없습니다: {args.vocab_path}\n"
            "먼저 `bash scripts/build_vocab_light.sh` 를 실행하세요."
        )
    if not args.finetuned_checkpoint.exists():
        raise SystemExit(
            f"finetune checkpoint가 없습니다: {args.finetuned_checkpoint}\n"
            "먼저 `bash scripts/run_finetune_light.sh` 를 실행하세요."
        )

    device = pick_device(args.device)
    print(f"device: {device}")

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.load(args.vocab_path)

    test_rows = read_jsonl(args.test_path)
    test_ds = ReviewSentimentDataset(test_rows, tokenizer, max_length=args.context_length)
    test_loader = make_dataloader(test_ds, args.batch_size, False, args.num_workers, device)

    config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias,
    }
    backbone = GPTModel(config)
    model = GPTForSequenceClassification(backbone, num_labels=2, drop_rate=args.drop_rate).to(device)

    load_finetuned_checkpoint(model, args.finetuned_checkpoint, device)

    eval_start = time.perf_counter()
    test_loss, test_acc = evaluate_sentiment(model, test_loader, device)
    test_elapsed_seconds = time.perf_counter() - eval_start

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(args.artifact_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test_loss": float(test_loss), "test_acc": float(test_acc)}, f, ensure_ascii=False, indent=2)
    write_timing_json(
        args.artifact_dir / "timing.json",
        {
            "test_elapsed_seconds": test_elapsed_seconds,
            "num_test_examples": len(test_rows),
            "num_workers": args.num_workers,
            "device": str(device),
            "samples_per_second": len(test_rows) / test_elapsed_seconds if test_elapsed_seconds > 0 else 0.0,
            "milliseconds_per_sample": (test_elapsed_seconds / len(test_rows) * 1000) if test_rows else 0.0,
        },
    )

    print(f"artifact_dir: {args.artifact_dir}")
    print(f"test loss: {test_loss:.3f}")
    print(f"test acc: {test_acc:.3f}")
    print(f"test_elapsed_seconds: {test_elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()
