# -*- coding: utf-8 -*-
"""NSMC Light baseline용 감성 분류 fine-tuning 러너."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe import BPETokenizer
from src.finetune import (
    GPTForSequenceClassification,
    ReviewSentimentDataset,
    evaluate_sentiment,
    train_step_sentiment,
)
from src.model import GPTModel
from src.train import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NSMC sentiment fine-tuning on Light baseline.")
    parser.add_argument("--train-path", type=Path, required=True, help="Sentiment train JSONL path.")
    parser.add_argument("--val-path", type=Path, required=True, help="Sentiment validation JSONL path.")
    parser.add_argument("--vocab-path", type=Path, required=True, help="Saved BPE vocab JSON path.")
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="artifacts/pretrain_light/checkpoints/best.pt",
        help="Pretrained GPT checkpoint path. Empty string means skip loading.",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Directory for metrics, plots, and checkpoints.")
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--qkv-bias", action="store_true", help="Enable QKV bias.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--eval-freq", type=int, default=1500, help="Evaluate and log every N train steps.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
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


def save_metric_plot(
    path: Path,
    x_values: list[int],
    train_values: list[float],
    val_values: list[float],
    ylabel: str,
    title: str,
    xlabel: str = "Global Step",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(x_values, train_values, label="Train")
    plt.plot(x_values, val_values, label="Val")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_metrics_jsonl(
    path: Path,
    epochs: list[int],
    global_steps: list[int],
    train_losses: list[float],
    train_accs: list[float],
    val_losses: list[float],
    val_accs: list[float],
    train_seconds: list[float],
    val_seconds: list[float],
    interval_seconds: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for epoch, step, train_loss, train_acc, val_loss, val_acc, train_time, val_time, interval_time in zip(
            epochs,
            global_steps,
            train_losses,
            train_accs,
            val_losses,
            val_accs,
            train_seconds,
            val_seconds,
            interval_seconds,
        ):
            row = {
                "epoch": epoch,
                "step": step,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "train_elapsed_seconds": float(train_time),
                "val_elapsed_seconds": float(val_time),
                "interval_elapsed_seconds": float(interval_time),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_classifier_checkpoint(path: Path, model: GPTForSequenceClassification, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
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

    pretrained_checkpoint = args.pretrained_checkpoint.strip()
    if pretrained_checkpoint and not Path(pretrained_checkpoint).exists():
        raise SystemExit(
            f"pretrain checkpoint가 없습니다: {pretrained_checkpoint}\n"
            "먼저 `bash scripts/run_pretrain_light.sh` 를 실행하세요."
        )

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.load(args.vocab_path)

    train_rows = read_jsonl(args.train_path)
    val_rows = read_jsonl(args.val_path)

    train_ds = ReviewSentimentDataset(train_rows, tokenizer, max_length=args.context_length)
    val_ds = ReviewSentimentDataset(val_rows, tokenizer, max_length=args.context_length)

    train_loader = make_dataloader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = make_dataloader(val_ds, args.batch_size, False, args.num_workers, device)

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
    if pretrained_checkpoint:
        load_checkpoint(backbone, None, pretrained_checkpoint, device)

    model = GPTForSequenceClassification(backbone, num_labels=2, drop_rate=args.drop_rate).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    artifact_dir = args.artifact_dir
    checkpoint_dir = artifact_dir / "checkpoints"
    eval_epochs: list[int] = []
    eval_steps: list[int] = []
    train_losses: list[float] = []
    train_accs: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []
    train_seconds: list[float] = []
    val_seconds: list[float] = []
    interval_seconds: list[float] = []
    best_val_loss = float("inf")
    global_step = 0
    last_epoch = 0
    interval_loss_sum = 0.0
    interval_correct = 0
    interval_count = 0
    interval_step_count = 0
    interval_train_start = time.perf_counter()

    def record_evaluation(epoch: int, step: int) -> None:
        nonlocal best_val_loss
        nonlocal interval_loss_sum
        nonlocal interval_correct
        nonlocal interval_count
        nonlocal interval_step_count
        nonlocal interval_train_start

        train_elapsed = time.perf_counter() - interval_train_start
        val_start = time.perf_counter()
        val_loss, val_acc = evaluate_sentiment(model, val_loader, device)
        val_elapsed = time.perf_counter() - val_start

        train_loss = interval_loss_sum / max(interval_step_count, 1)
        train_acc = interval_correct / max(interval_count, 1)

        eval_epochs.append(epoch)
        eval_steps.append(step)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        train_seconds.append(train_elapsed)
        val_seconds.append(val_elapsed)
        interval_seconds.append(train_elapsed + val_elapsed)

        print(
            f"epoch {epoch:02d} (step: {step:06d}): "
            f"train loss {train_loss:.3f}, train acc {train_acc:.3f}, "
            f"val loss {val_loss:.3f}, val acc {val_acc:.3f}, "
            f"interval time {train_elapsed + val_elapsed:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_classifier_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch)

        interval_loss_sum = 0.0
        interval_correct = 0
        interval_count = 0
        interval_step_count = 0
        interval_train_start = time.perf_counter()

    for epoch in range(1, args.num_epochs + 1):
        last_epoch = epoch
        for input_ids, labels in train_loader:
            loss_value, batch_correct, batch_count = train_step_sentiment(model, input_ids, labels, optimizer, device)
            global_step += 1
            interval_loss_sum += loss_value
            interval_correct += batch_correct
            interval_count += batch_count
            interval_step_count += 1

            if global_step % args.eval_freq == 0:
                record_evaluation(epoch, global_step)

    if interval_step_count > 0:
        record_evaluation(last_epoch, global_step)

    if last_epoch > 0:
        save_classifier_checkpoint(checkpoint_dir / "last.pt", model, optimizer, last_epoch)

    fit_elapsed_seconds = time.perf_counter() - run_start
    save_metrics_jsonl(
        artifact_dir / "metrics.jsonl",
        eval_epochs,
        eval_steps,
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        train_seconds,
        val_seconds,
        interval_seconds,
    )
    save_metric_plot(
        artifact_dir / "loss.png",
        eval_steps,
        train_losses,
        val_losses,
        ylabel="Loss",
        title="Sentiment Loss",
    )
    save_metric_plot(
        artifact_dir / "accuracy.png",
        eval_steps,
        train_accs,
        val_accs,
        ylabel="Accuracy",
        title="Sentiment Accuracy",
    )
    total_train_examples = len(train_rows) * args.num_epochs
    write_timing_json(
        artifact_dir / "timing.json",
        {
            "fit_elapsed_seconds": fit_elapsed_seconds,
            "train_phase_elapsed_seconds": sum(train_seconds),
            "val_phase_elapsed_seconds": sum(val_seconds),
            "total_train_examples": total_train_examples,
            "total_train_steps": global_step,
            "eval_freq": args.eval_freq,
            "num_workers": args.num_workers,
            "device": str(device),
            "num_eval_points": len(eval_steps),
            "train_examples_per_second": total_train_examples / sum(train_seconds) if sum(train_seconds) > 0 else 0.0,
        },
    )

    print(f"artifact_dir: {artifact_dir}")
    print(f"best_checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"last_checkpoint: {checkpoint_dir / 'last.pt'}")
    print(f"fit_elapsed_seconds: {fit_elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()
