# -*- coding: utf-8 -*-
"""발표용 실험 로그 요약과 노트북 구조를 검증한다."""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestExperimentLog:
    def test_summarize_pretrain_run_collects_best_and_last_metrics(self):
        from experiment_log import summarize_pretrain_run

        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "pretrain"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            _write_jsonl(
                artifact_dir / "metrics.jsonl",
                [
                    {"step": 50, "tokens_seen": 100, "train_loss": 2.3, "val_loss": 2.5},
                    {"step": 100, "tokens_seen": 200, "train_loss": 2.1, "val_loss": 2.2},
                    {"step": 150, "tokens_seen": 300, "train_loss": 1.9, "val_loss": 2.4},
                ],
            )
            (artifact_dir / "timing.json").write_text(
                json.dumps({"train_elapsed_seconds": 12.5}, ensure_ascii=False),
                encoding="utf-8",
            )

            summary = summarize_pretrain_run(
                run_name="light_a",
                artifact_dir=artifact_dir,
                changed_hyperparam="learning_rate",
                changed_value="3e-4",
                note="baseline",
                next_action="batch_size",
            )

            assert summary["stage"] == "pretrain"
            assert summary["train_loss"] == 1.9
            assert summary["val_loss"] == 2.2
            assert summary["best_val_loss"] == 2.2
            assert summary["last_train_loss"] == 1.9
            assert math.isclose(summary["val_ppl"], math.exp(2.2), rel_tol=1e-9)
            assert summary["train_elapsed_seconds"] == 12.5

    def test_summarize_finetune_run_collects_best_val_epoch(self):
        from experiment_log import summarize_finetune_run

        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "finetune"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            _write_jsonl(
                artifact_dir / "metrics.jsonl",
                [
                    {"epoch": 1, "train_loss": 0.70, "train_acc": 0.60, "val_loss": 0.68, "val_acc": 0.62},
                    {"epoch": 2, "train_loss": 0.63, "train_acc": 0.68, "val_loss": 0.61, "val_acc": 0.70},
                    {"epoch": 3, "train_loss": 0.59, "train_acc": 0.72, "val_loss": 0.66, "val_acc": 0.67},
                ],
            )
            (artifact_dir / "timing.json").write_text(
                json.dumps({"fit_elapsed_seconds": 8.0}, ensure_ascii=False),
                encoding="utf-8",
            )

            summary = summarize_finetune_run("ft_a", artifact_dir, "drop_rate", "0.1", "baseline", "freeze")

            assert summary["stage"] == "finetune"
            assert summary["train_loss"] == 0.59
            assert summary["val_loss"] == 0.61
            assert summary["best_val_loss"] == 0.61
            assert summary["last_train_loss"] == 0.59
            assert summary["best_val_acc"] == 0.70
            assert summary["fit_elapsed_seconds"] == 8.0

    def test_summarize_pretrain_run_resolves_relative_path_from_repo_root(self):
        from experiment_log import summarize_pretrain_run

        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            artifact_dir = ROOT / "artifacts" / "relative_path_case" / "pretrain"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            _write_jsonl(
                artifact_dir / "metrics.jsonl",
                [
                    {"step": 50, "tokens_seen": 100, "train_loss": 2.3, "val_loss": 2.5},
                    {"step": 100, "tokens_seen": 200, "train_loss": 2.1, "val_loss": 2.2},
                ],
            )
            (artifact_dir / "timing.json").write_text(
                json.dumps({"train_elapsed_seconds": 12.5}, ensure_ascii=False),
                encoding="utf-8",
            )

            try:
                os.chdir(tmp)
                summary = summarize_pretrain_run("light_rel", "artifacts/relative_path_case/pretrain")
            finally:
                os.chdir(original_cwd)
                for path in [artifact_dir / "metrics.jsonl", artifact_dir / "timing.json"]:
                    if path.exists():
                        path.unlink()
                if artifact_dir.exists():
                    artifact_dir.rmdir()
                parent = ROOT / "artifacts" / "relative_path_case"
                if parent.exists():
                    parent.rmdir()
                artifacts_root = ROOT / "artifacts"
                if artifacts_root.exists() and not any(artifacts_root.iterdir()):
                    artifacts_root.rmdir()

            assert summary["stage"] == "pretrain"
            assert summary["val_loss"] == 2.2

    def test_summarize_pretrain_run_missing_file_has_actionable_message(self):
        from experiment_log import summarize_pretrain_run

        with tempfile.TemporaryDirectory() as tmp:
            missing_dir = Path(tmp) / "pretrain"

            try:
                summarize_pretrain_run("missing", missing_dir)
                assert False, "Expected FileNotFoundError"
            except FileNotFoundError as exc:
                message = str(exc)
                assert "pretrain metrics 파일이 없습니다" in message
                assert "run_pretrain_light.sh" in message
                assert "현재 기준 경로" in message

    def test_upsert_experiment_log_overwrites_same_run_stage(self):
        from experiment_log import load_experiment_log_rows, upsert_experiment_log

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "experiment_log.csv"
            upsert_experiment_log(
                csv_path,
                {
                    "run": "light_a",
                    "stage": "pretrain",
                    "changed_hyperparam": "learning_rate",
                    "changed_value": "3e-4",
                    "train_loss": 2.1,
                    "val_loss": 2.2,
                },
            )
            upsert_experiment_log(
                csv_path,
                {
                    "run": "light_a",
                    "stage": "pretrain",
                    "changed_hyperparam": "learning_rate",
                    "changed_value": "1e-4",
                    "train_loss": 2.0,
                    "val_loss": 2.1,
                },
            )
            rows = load_experiment_log_rows(csv_path)

            assert len(rows) == 1
            assert rows[0]["run"] == "light_a"
            assert rows[0]["stage"] == "pretrain"
            assert rows[0]["changed_value"] == "1e-4"
            assert rows[0]["val_loss"] == "2.1"


class TestExperimentNotebook:
    def test_notebook_has_summary_cells_and_guidance(self):
        notebook = json.loads((ROOT / "gpt-lab.ipynb").read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        joined = "\n".join(sources)

        assert "### 10.6 실험 로그 요약 셀" in joined
        assert "### 10.7 누적 실험 표 보기 셀" in joined
        assert "tokenizer 조건이 바뀌면 token-level loss 직접 비교가 공정하지 않을 수 있습니다." in joined
        assert "# 이번 run에서는 baseline 대비 바꾼 항목을 한 가지만 유지하세요." in joined
        assert "STRIDE = CONTEXT_LENGTH" in joined
        assert "VAL_SPLIT_RATIO = 0.08" in joined
