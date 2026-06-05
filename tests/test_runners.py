# -*- coding: utf-8 -*-
"""실험용 runner/셸 인터페이스 smoke test."""

import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestRunnerCli:
    """새 runner가 --help 기준으로 실행 가능한지 확인."""

    def test_python_runners_show_help(self):
        for script_name in ["run_build_vocab.py", "run_pretrain.py", "run_finetune.py", "run_test_eval.py"]:
            result = subprocess.run(
                [sys.executable, str(ROOT / script_name), "--help"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            assert "usage:" in result.stdout.lower()

    def test_shell_wrappers_are_executable(self):
        for script_name in [
            "prepare_nsmc.sh",
            "build_vocab_light.sh",
            "run_pretrain_light.sh",
            "run_finetune_light.sh",
            "run_test_eval_light.sh",
        ]:
            path = ROOT / "scripts" / script_name
            mode = path.stat().st_mode
            assert stat.S_ISREG(mode)
            assert os.access(path, os.X_OK)


class TestRunnerWiring:
    """설정 값이 실제 runner 내부 인자에 반영되는지 확인."""

    def test_run_pretrain_uses_cli_hyperparameters(self):
        import run_pretrain

        captured = {}

        class DummyTokenizer:
            def __init__(self, vocab_size):
                captured["tokenizer_vocab_size"] = vocab_size

            def load(self, path):
                captured["vocab_path"] = str(path)

            def encode(self, text):
                return [1, 2, 3, 4, 5, 6]

        class DummyModel:
            def __init__(self, config):
                captured["model_config"] = config

            def to(self, device):
                captured["device"] = str(device)
                return self

            def parameters(self):
                return []

        def fake_dataloader(token_ids, context_length, batch_size, stride, drop_last, shuffle, num_workers=0):
            captured.setdefault("dataloaders", []).append(
                {
                    "context_length": context_length,
                    "batch_size": batch_size,
                    "stride": stride,
                    "drop_last": drop_last,
                    "shuffle": shuffle,
                }
            )
            return ["loader"]

        def fake_train_model(**kwargs):
            captured["train_model_args"] = kwargs
            return [0.5], [0.6], [128], [{"epoch": 1, "global_step": 1, "sample_text": "sample"}]

        def fake_plot_losses(x_values, train_losses, val_losses, output_path=None):
            captured["plot_output_path"] = str(output_path)

        train_path = ROOT / "tests" / "tmp_train.txt"
        val_path = ROOT / "tests" / "tmp_val.txt"
        vocab_path = ROOT / "tests" / "tmp_vocab.json"
        train_path.write_text("abc", encoding="utf-8")
        val_path.write_text("def", encoding="utf-8")
        vocab_path.write_text("{}", encoding="utf-8")

        with mock.patch.object(
            run_pretrain,
            "parse_args",
            return_value=SimpleNamespace(
                train_path=train_path,
                val_path=val_path,
                vocab_path=vocab_path,
                artifact_dir=ROOT / "tests" / "tmp_artifacts",
                train_char_limit=1234,
                val_char_limit=4321,
                vocab_size=2222,
                context_length=77,
                stride=33,
                emb_dim=88,
                n_heads=3,
                n_layers=4,
                drop_rate=0.25,
                qkv_bias=True,
                batch_size=9,
                learning_rate=1e-4,
                weight_decay=0.2,
                num_epochs=2,
                eval_freq=7,
                eval_iter=3,
                ckpt_freq=5,
                start_context="테스트",
                seed=42,
                device="cpu",
            ),
        ), mock.patch.object(run_pretrain, "BPETokenizer", DummyTokenizer), mock.patch.object(
            run_pretrain, "create_dataloader", side_effect=fake_dataloader
        ), mock.patch.object(run_pretrain, "GPTModel", DummyModel), mock.patch.object(
            run_pretrain.torch.optim, "AdamW", return_value=object()
        ), mock.patch.object(
            run_pretrain, "train_model", side_effect=fake_train_model
        ), mock.patch.object(
            run_pretrain, "plot_losses", side_effect=fake_plot_losses
        ), mock.patch.object(
            run_pretrain, "write_metrics_jsonl"
        ), mock.patch.object(run_pretrain, "write_samples"), mock.patch.object(
            run_pretrain, "write_timing_json"
        ) as timing_writer:
            run_pretrain.main()

        assert captured["model_config"]["vocab_size"] == 2222
        assert captured["model_config"]["context_length"] == 77
        assert captured["model_config"]["emb_dim"] == 88
        assert captured["model_config"]["n_layers"] == 4
        assert captured["model_config"]["drop_rate"] == 0.25
        assert captured["model_config"]["qkv_bias"] is True
        assert captured["dataloaders"][0]["batch_size"] == 9
        assert captured["dataloaders"][0]["context_length"] == 77
        assert captured["dataloaders"][0]["stride"] == 33
        assert captured["train_model_args"]["num_epochs"] == 2
        assert captured["train_model_args"]["eval_freq"] == 7
        assert captured["plot_output_path"].endswith("tmp_artifacts/loss.png")
        assert timing_writer.called

        train_path.unlink(missing_ok=True)
        val_path.unlink(missing_ok=True)
        vocab_path.unlink(missing_ok=True)

    def test_run_finetune_uses_cli_hyperparameters(self):
        import run_finetune

        captured = {"train_step_calls": 0}

        class DummyTokenizer:
            def __init__(self, vocab_size):
                captured["tokenizer_vocab_size"] = vocab_size

            def load(self, path):
                captured["vocab_path"] = str(path)

        class DummyBackbone:
            def __init__(self, config):
                captured["model_config"] = config

        class DummyClassifier:
            def __init__(self, backbone, num_labels, drop_rate, unfreeze_backbone=False):
                captured["classifier"] = {
                    "num_labels": num_labels,
                    "drop_rate": drop_rate,
                    "unfreeze_backbone": unfreeze_backbone,
                }

            def to(self, device):
                captured["device"] = str(device)
                return self

            def named_parameters(self):
                return []

        def fake_dataset(rows, tokenizer, max_length):
            captured.setdefault("dataset_max_lengths", []).append(max_length)
            return rows

        def fake_dataloader(dataset, **kwargs):
            captured.setdefault("loaders", []).append(kwargs)
            return [([1, 2, 3], [1])]

        def fake_train_step(model, input_ids, labels, optimizer, device):
            captured["train_step_calls"] += 1
            return 0.4, 1, 1

        def fake_evaluate(model, loader, device):
            captured["evaluate_calls"] = captured.get("evaluate_calls", 0) + 1
            return 0.5, 0.7

        def fake_create_optimizer(model, classifier_lr, backbone_lr_ratio, weight_decay):
            captured["optimizer_config"] = {
                "classifier_lr": classifier_lr,
                "backbone_lr_ratio": backbone_lr_ratio,
                "weight_decay": weight_decay,
            }
            return object()

        def fake_summarize_trainable_params(model):
            return 128, 64

        def fake_read_jsonl(path):
            return [{"text": "sample", "label": 1}]

        vocab_path = ROOT / "tests" / "sent_vocab.json"
        checkpoint_path = ROOT / "tests" / "sent_best.pt"
        vocab_path.write_text("{}", encoding="utf-8")
        checkpoint_path.write_text("{}", encoding="utf-8")

        with mock.patch.object(
            run_finetune,
            "parse_args",
            return_value=SimpleNamespace(
                train_path=ROOT / "tests" / "sent_train.jsonl",
                val_path=ROOT / "tests" / "sent_val.jsonl",
                vocab_path=vocab_path,
                pretrained_checkpoint=str(checkpoint_path),
                artifact_dir=ROOT / "tests" / "sent_artifacts",
                vocab_size=3333,
                context_length=55,
                emb_dim=66,
                n_heads=7,
                n_layers=2,
                drop_rate=0.35,
                qkv_bias=True,
                batch_size=11,
                learning_rate=2e-5,
                weight_decay=0.15,
                num_epochs=4,
                eval_freq=2,
                num_workers=2,
                unfreeze_backbone=True,
                backbone_lr_ratio=0.1,
                seed=7,
                device="cpu",
            ),
        ), mock.patch.object(run_finetune, "BPETokenizer", DummyTokenizer), mock.patch.object(
            run_finetune, "read_jsonl", side_effect=fake_read_jsonl
        ), mock.patch.object(
            run_finetune, "ReviewSentimentDataset", side_effect=fake_dataset
        ), mock.patch.object(
            run_finetune, "DataLoader", side_effect=fake_dataloader
        ), mock.patch.object(
            run_finetune, "GPTModel", DummyBackbone
        ), mock.patch.object(
            run_finetune, "load_checkpoint"
        ), mock.patch.object(
            run_finetune, "GPTForSequenceClassification", DummyClassifier
        ), mock.patch.object(
            run_finetune, "create_optimizer", side_effect=fake_create_optimizer
        ), mock.patch.object(
            run_finetune, "summarize_trainable_params", side_effect=fake_summarize_trainable_params
        ), mock.patch.object(
            run_finetune, "train_step_sentiment", side_effect=fake_train_step
        ), mock.patch.object(
            run_finetune, "evaluate_sentiment", side_effect=fake_evaluate
        ), mock.patch.object(
            run_finetune, "save_classifier_checkpoint"
        ), mock.patch.object(
            run_finetune, "save_metrics_jsonl"
        ), mock.patch.object(
            run_finetune, "save_metric_plot"
        ), mock.patch.object(
            run_finetune, "write_timing_json"
        ) as timing_writer:
            run_finetune.main()

        assert captured["model_config"]["vocab_size"] == 3333
        assert captured["model_config"]["context_length"] == 55
        assert captured["model_config"]["emb_dim"] == 66
        assert captured["model_config"]["n_heads"] == 7
        assert captured["model_config"]["n_layers"] == 2
        assert captured["model_config"]["drop_rate"] == 0.35
        assert captured["model_config"]["qkv_bias"] is True
        assert captured["dataset_max_lengths"] == [55, 55]
        assert captured["loaders"][0]["batch_size"] == 11
        assert captured["loaders"][0]["num_workers"] == 2
        assert captured["loaders"][0]["persistent_workers"] is True
        assert captured["loaders"][0]["pin_memory"] is False
        assert captured["classifier"]["drop_rate"] == 0.35
        assert captured["classifier"]["unfreeze_backbone"] is True
        assert captured["optimizer_config"]["classifier_lr"] == 2e-5
        assert captured["optimizer_config"]["backbone_lr_ratio"] == 0.1
        assert captured["optimizer_config"]["weight_decay"] == 0.15
        assert captured["train_step_calls"] == 4
        assert captured["evaluate_calls"] == 2
        assert timing_writer.called

        vocab_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)

    def test_run_test_eval_uses_cli_hyperparameters(self):
        import run_test_eval

        captured = {}

        class DummyTokenizer:
            def __init__(self, vocab_size):
                captured["tokenizer_vocab_size"] = vocab_size

            def load(self, path):
                captured["vocab_path"] = str(path)

        class DummyBackbone:
            def __init__(self, config):
                captured["model_config"] = config

        class DummyClassifier:
            def __init__(self, backbone, num_labels, drop_rate):
                captured["classifier"] = {
                    "num_labels": num_labels,
                    "drop_rate": drop_rate,
                }

            def to(self, device):
                captured["device"] = str(device)
                return self

            def load_state_dict(self, state_dict, strict=False):
                captured["loaded_state_dict"] = state_dict
                captured["load_state_dict_strict"] = strict
                return [], []

        def fake_dataset(rows, tokenizer, max_length):
            captured["dataset_max_length"] = max_length
            return rows

        def fake_dataloader(dataset, **kwargs):
            captured["loader"] = kwargs
            return [dataset]

        def fake_evaluate(model, loader, device):
            captured["evaluate_called"] = True
            return 0.12, 0.91

        def fake_read_jsonl(path):
            captured["test_path"] = str(path)
            return [{"text": "sample", "label": 1}]

        vocab_path = ROOT / "tests" / "eval_vocab.json"
        checkpoint_path = ROOT / "tests" / "eval_best.pt"
        artifact_dir = ROOT / "tests" / "eval_artifacts"
        vocab_path.write_text("{}", encoding="utf-8")
        checkpoint_path.write_text("{}", encoding="utf-8")

        with mock.patch.object(
            run_test_eval,
            "parse_args",
            return_value=SimpleNamespace(
                test_path=ROOT / "tests" / "sent_test.jsonl",
                vocab_path=vocab_path,
                finetuned_checkpoint=checkpoint_path,
                artifact_dir=artifact_dir,
                vocab_size=4444,
                context_length=48,
                emb_dim=96,
                n_heads=6,
                n_layers=3,
                drop_rate=0.2,
                qkv_bias=True,
                batch_size=13,
                num_workers=2,
                device="cpu",
            ),
        ), mock.patch.object(run_test_eval, "BPETokenizer", DummyTokenizer), mock.patch.object(
            run_test_eval, "read_jsonl", side_effect=fake_read_jsonl
        ), mock.patch.object(
            run_test_eval, "ReviewSentimentDataset", side_effect=fake_dataset
        ), mock.patch.object(
            run_test_eval, "DataLoader", side_effect=fake_dataloader
        ), mock.patch.object(
            run_test_eval, "GPTModel", DummyBackbone
        ), mock.patch.object(
            run_test_eval, "GPTForSequenceClassification", DummyClassifier
        ), mock.patch.object(
            run_test_eval, "evaluate_sentiment", side_effect=fake_evaluate
        ), mock.patch.object(
            run_test_eval.torch, "load", return_value={"model_state_dict": {"ok": 1, "gpt.trf_blocks.0.att.mask": 1}}
        ), mock.patch.object(
            run_test_eval, "write_timing_json"
        ) as timing_writer:
            run_test_eval.main()

        assert captured["model_config"]["vocab_size"] == 4444
        assert captured["model_config"]["context_length"] == 48
        assert captured["model_config"]["emb_dim"] == 96
        assert captured["model_config"]["n_heads"] == 6
        assert captured["model_config"]["n_layers"] == 3
        assert captured["model_config"]["qkv_bias"] is True
        assert captured["dataset_max_length"] == 48
        assert captured["loader"]["batch_size"] == 13
        assert captured["loader"]["shuffle"] is False
        assert captured["loader"]["num_workers"] == 2
        assert captured["loader"]["persistent_workers"] is True
        assert captured["loader"]["pin_memory"] is False
        assert captured["classifier"]["drop_rate"] == 0.2
        assert captured["load_state_dict_strict"] is False
        assert "gpt.trf_blocks.0.att.mask" not in captured["loaded_state_dict"]
        assert captured["evaluate_called"] is True
        assert (artifact_dir / "test_metrics.json").exists()
        assert timing_writer.called

        vocab_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
        (artifact_dir / "test_metrics.json").unlink(missing_ok=True)
        artifact_dir.rmdir()
