# -*- coding: utf-8 -*-
"""NSMC Light baseline용 GPT 사전학습 러너."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe import BPETokenizer
from src.dataset import create_dataloader
from src.model import GPTModel
from src.train import plot_losses, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT pretraining on NSMC Light baseline.")
    parser.add_argument("--train-path", type=Path, required=True, help="LM train text path.")
    parser.add_argument("--val-path", type=Path, required=True, help="LM validation text path.")
    parser.add_argument("--vocab-path", type=Path, required=True, help="Saved BPE vocab JSON path.")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Directory for metrics, plots, and checkpoints.")
    parser.add_argument("--train-char-limit", type=int, default=500000, help="Train corpus character limit.")
    parser.add_argument("--val-char-limit", type=int, default=0, help="Validation corpus character limit. 0 means full file.")
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=None, help="Sliding window stride. Defaults to context_length.")
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--qkv-bias", action="store_true", help="Enable QKV bias.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=50)
    parser.add_argument("--eval-iter", type=int, default=10)
    parser.add_argument("--ckpt-freq", type=int, default=200)
    parser.add_argument("--start-context", type=str, default="이 영화는")
    parser.add_argument("--seed", type=int, default=123)
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


def read_corpus(path: Path, char_limit: int) -> str:
    if not path.exists():
        raise SystemExit(f"LM text가 없습니다: {path}")
    text = path.read_text(encoding="utf-8")
    if char_limit > 0:
        return text[:char_limit]
    return text


def write_metrics_jsonl(path: Path, steps: list[int], tokens_seen: list[int], train_losses: list[float], val_losses: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for step, token_count, train_loss, val_loss in zip(steps, tokens_seen, train_losses, val_losses):
            row = {
                "step": step,
                "tokens_seen": token_count,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_samples(path: Path, sample_texts: list[dict[str, int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in sample_texts:
            f.write(
                f"epoch={sample['epoch']} "
                f"global_step={sample['global_step']} "
                f"sample_text={sample['sample_text']}\n"
            )


def write_timing_json(path: Path, timing: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timing, f, ensure_ascii=False, indent=2)


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    if not args.vocab_path.exists():
        raise SystemExit(
            f"vocab 파일이 없습니다: {args.vocab_path}\n"
            "먼저 `bash scripts/build_vocab_light.sh` 를 실행하세요."
        )

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    train_corpus = read_corpus(args.train_path, args.train_char_limit)
    val_corpus = read_corpus(args.val_path, args.val_char_limit)
    if not train_corpus or not val_corpus:
        raise SystemExit("train/val corpus가 비어 있어 pretrain을 시작할 수 없습니다.")

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.load(args.vocab_path)

    train_ids = tokenizer.encode(train_corpus)
    val_ids = tokenizer.encode(val_corpus)
    stride = args.context_length if args.stride is None else args.stride

    train_loader = create_dataloader(
        train_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=stride,
        drop_last=True,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=stride,
        drop_last=False,
        shuffle=False,
    )

    config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias,
    }
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    artifact_dir = args.artifact_dir
    checkpoint_dir = artifact_dir / "checkpoints"
    train_start = time.perf_counter()
    train_losses, val_losses, tokens_seen, sample_texts = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        start_context=args.start_context,
        tokenizer=tokenizer,
        ckpt_freq=args.ckpt_freq,
        checkpoint_dir=checkpoint_dir,
    )
    train_elapsed_seconds = time.perf_counter() - train_start
    total_elapsed_seconds = time.perf_counter() - run_start

    steps = [args.eval_freq * (idx + 1) for idx in range(len(train_losses))]
    write_metrics_jsonl(artifact_dir / "metrics.jsonl", steps, tokens_seen, train_losses, val_losses)
    write_samples(artifact_dir / "samples.txt", sample_texts)
    plot_losses(tokens_seen, train_losses, val_losses, output_path=artifact_dir / "loss.png")
    final_tokens_seen = tokens_seen[-1] if tokens_seen else 0
    write_timing_json(
        artifact_dir / "timing.json",
        {
            "train_elapsed_seconds": train_elapsed_seconds,
            "total_elapsed_seconds": total_elapsed_seconds,
            "tokens_seen": final_tokens_seen,
            "tokens_per_second": final_tokens_seen / train_elapsed_seconds if train_elapsed_seconds > 0 else 0.0,
        },
    )

    print(f"artifact_dir: {artifact_dir}")
    print(f"best_checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"last_checkpoint: {checkpoint_dir / 'last.pt'}")
    print(f"train_elapsed_seconds: {train_elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()
