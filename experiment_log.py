# -*- coding: utf-8 -*-
"""발표용 실험 로그 요약과 CSV 누적을 돕는 유틸리티."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

EXPERIMENT_LOG_COLUMNS = [
    "run",
    "stage",
    "changed_hyperparam",
    "changed_value",
    "train_loss",
    "val_loss",
    "note",
    "next_action",
    "best_val_loss",
    "last_train_loss",
    "val_ppl",
    "train_elapsed_seconds",
    "best_val_acc",
    "fit_elapsed_seconds",
]

NUMERIC_FIELDS = {
    "train_loss",
    "val_loss",
    "best_val_loss",
    "last_train_loss",
    "val_ppl",
    "train_elapsed_seconds",
    "best_val_acc",
    "fit_elapsed_seconds",
}


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} 파일이 없습니다: {path}")


def summarize_pretrain_run(
    run_name: str,
    artifact_dir: str | Path,
    changed_hyperparam: str = "",
    changed_value: str = "",
    note: str = "",
    next_action: str = "",
) -> dict[str, str | float]:
    artifact_dir = Path(artifact_dir)
    metrics_path = artifact_dir / "metrics.jsonl"
    timing_path = artifact_dir / "timing.json"
    _ensure_file(metrics_path, "pretrain metrics")
    _ensure_file(timing_path, "pretrain timing")

    metrics = _read_jsonl(metrics_path)
    if not metrics:
        raise ValueError(f"pretrain metrics가 비어 있습니다: {metrics_path}")
    timing = _read_json(timing_path)

    best_row = min(metrics, key=lambda row: float(row["val_loss"]))
    last_row = metrics[-1]
    best_val_loss = float(best_row["val_loss"])
    last_train_loss = float(last_row["train_loss"])

    return {
        "run": run_name,
        "stage": "pretrain",
        "changed_hyperparam": changed_hyperparam,
        "changed_value": changed_value,
        "train_loss": last_train_loss,
        "val_loss": best_val_loss,
        "note": note,
        "next_action": next_action,
        "best_val_loss": best_val_loss,
        "last_train_loss": last_train_loss,
        "val_ppl": math.exp(best_val_loss),
        "train_elapsed_seconds": float(timing.get("train_elapsed_seconds", 0.0)),
        "best_val_acc": "",
        "fit_elapsed_seconds": "",
    }


def summarize_finetune_run(
    run_name: str,
    artifact_dir: str | Path,
    changed_hyperparam: str = "",
    changed_value: str = "",
    note: str = "",
    next_action: str = "",
) -> dict[str, str | float]:
    artifact_dir = Path(artifact_dir)
    metrics_path = artifact_dir / "metrics.jsonl"
    timing_path = artifact_dir / "timing.json"
    _ensure_file(metrics_path, "finetune metrics")
    _ensure_file(timing_path, "finetune timing")

    metrics = _read_jsonl(metrics_path)
    if not metrics:
        raise ValueError(f"finetune metrics가 비어 있습니다: {metrics_path}")
    timing = _read_json(timing_path)

    best_row = min(metrics, key=lambda row: float(row["val_loss"]))
    last_row = metrics[-1]
    best_val_loss = float(best_row["val_loss"])
    last_train_loss = float(last_row["train_loss"])

    return {
        "run": run_name,
        "stage": "finetune",
        "changed_hyperparam": changed_hyperparam,
        "changed_value": changed_value,
        "train_loss": last_train_loss,
        "val_loss": best_val_loss,
        "note": note,
        "next_action": next_action,
        "best_val_loss": best_val_loss,
        "last_train_loss": last_train_loss,
        "val_ppl": "",
        "train_elapsed_seconds": "",
        "best_val_acc": float(best_row["val_acc"]),
        "fit_elapsed_seconds": float(timing.get("fit_elapsed_seconds", 0.0)),
    }


def load_experiment_log_rows(csv_path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def upsert_experiment_log(csv_path: str | Path, row: dict[str, str | float]) -> list[dict[str, str]]:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_row = {column: row.get(column, "") for column in EXPERIMENT_LOG_COLUMNS}
    normalized_row = {key: "" if value is None else value for key, value in normalized_row.items()}

    rows = load_experiment_log_rows(csv_path)
    rows = [existing for existing in rows if not (existing.get("run") == str(normalized_row["run"]) and existing.get("stage") == str(normalized_row["stage"]))]
    rows.append({key: str(value) for key, value in normalized_row.items()})

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return rows
