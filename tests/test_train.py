# -*- coding: utf-8 -*-
"""
사전 훈련 유틸 단위 테스트 (calc_loss_batch, calc_loss_loader, save/load_checkpoint, generate).
실행: `pytest tests/test_train.py -v`
"""

import sys
import tempfile
from pathlib import Path

import torch
import pytest
import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

GPT_CONFIG_SMALL = {
    "vocab_size": 1000,
    "context_length": 64,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


class TinyTokenizer:
    """train_model smoke test용 아주 작은 tokenizer."""

    def encode(self, text, add_bos_eos=False):
        ids = [(ord(ch) % 50) + 4 for ch in text]
        if add_bos_eos:
            return [2] + ids + [3]
        return ids

    def decode(self, ids):
        return " ".join(str(token_id) for token_id in ids)

    def get_eos_id(self):
        return 3


class RecoveringTokenizer(TinyTokenizer):
    """첫 decode는 실패하고, errors='replace'가 오면 복구하는 tokenizer."""

    def decode(self, ids, errors="strict"):
        if errors == "strict":
            raise UnicodeDecodeError("utf-8", b"\x8c", 0, 1, "invalid start byte")
        return "decoded with replacement"


# =============================================================================
# calc_loss_batch
# =============================================================================


class TestCalcLossBatch:
    """calc_loss_batch 구현 후 실행."""

    def test_calc_loss_batch_returns_scalar(self):
        """calc_loss_batch()가 한 배치의 cross entropy loss를 scalar Tensor로 반환하는지 확인한다."""
        from model import GPTModel
        from train import calc_loss_batch

        try:
            model = GPTModel(GPT_CONFIG_SMALL)
            device = torch.device("cpu")
            batch_size, seq_len = 2, 8
            inp = torch.randint(0, GPT_CONFIG_SMALL["vocab_size"], (batch_size, seq_len))
            tgt = torch.randint(0, GPT_CONFIG_SMALL["vocab_size"], (batch_size, seq_len))
            loss = calc_loss_batch(inp, tgt, model, device)
        except NotImplementedError:
            pytest.fail("calc_loss_batch 미구현")
        assert loss.dim() == 0
        assert loss.item() >= 0


# =============================================================================
# calc_loss_loader
# =============================================================================


class TestCalcLossLoader:
    """calc_loss_loader 구현 후 실행."""

    def test_calc_loss_loader_returns_float(self):
        """calc_loss_loader()가 여러 배치의 평균 손실을 float 또는 scalar Tensor로 반환하는지 확인한다."""
        from model import GPTModel
        from train import calc_loss_loader
        from dataset import create_dataloader

        try:
            loader = create_dataloader(
                list(range(200)),
                context_length=16,
                batch_size=4,
                shuffle=False,
            )
            model = GPTModel(GPT_CONFIG_SMALL)
            device = torch.device("cpu")
            avg_loss = calc_loss_loader(loader, model, device, num_batches=2)
        except NotImplementedError:
            pytest.fail("calc_loss_loader 또는 create_dataloader 미구현")
        assert isinstance(avg_loss, (float, torch.Tensor))
        if isinstance(avg_loss, torch.Tensor):
            avg_loss = avg_loss.item()
        assert avg_loss >= 0


# =============================================================================
# save_checkpoint / load_checkpoint
# =============================================================================


class TestCheckpoint:
    """save_checkpoint, load_checkpoint 구현 후 실행."""

    def test_save_load_checkpoint_restores_epoch_and_step(self):
        """체크포인트 저장 후 다른 모델/옵티마이저에 epoch와 step까지 복원되는지 확인한다."""
        from model import GPTModel
        from train import save_checkpoint, load_checkpoint

        try:
            model = GPTModel(GPT_CONFIG_SMALL)
            params = list(model.parameters())
            if not params:
                pytest.fail("GPTModel.__init__ 미구현")
            optimizer = torch.optim.AdamW(params, lr=1e-4)
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                path = f.name
            save_checkpoint(model, optimizer, epoch=1, global_step=10, path=path)
            model2 = GPTModel(GPT_CONFIG_SMALL)
            opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
            epoch, step = load_checkpoint(model2, opt2, path, torch.device("cpu"))
            Path(path).unlink(missing_ok=True)
        except NotImplementedError:
            pytest.fail("save_checkpoint/load_checkpoint 미구현")
        assert epoch == 1
        assert step == 10


# =============================================================================
# generate (temperature, top_k)
# =============================================================================


class TestGenerate:
    """generate 구현 후 실행."""

    def test_generate_shape(self):
        """generate()가 temperature/top_k 옵션을 사용해 max_new_tokens만큼 토큰을 추가하는지 확인한다."""
        from model import GPTModel
        from train import generate

        try:
            model = GPTModel(GPT_CONFIG_SMALL)
            idx = torch.randint(0, GPT_CONFIG_SMALL["vocab_size"], (1, 4))
            out = generate(
                model, idx, max_new_tokens=5, context_size=GPT_CONFIG_SMALL["context_length"],
                temperature=1.0, top_k=10,
            )
        except NotImplementedError:
            pytest.fail("generate 미구현")
        assert out.shape == (1, 4 + 5)


# =============================================================================
# plot_losses (제공 함수)
# =============================================================================


class TestPlotLosses:
    """plot_losses는 제공됨. 호출만 확인."""

    def test_plot_losses_callable(self):
        """plot_losses()가 train/val loss 리스트를 받아 예외 없이 그래프를 그리는지 확인한다."""
        from train import plot_losses

        plot_losses([0.5, 0.4, 0.3], [0.6, 0.5, 0.4])
        # 시각화만 하므로 예외 없으면 통과

    def test_plot_losses_saves_png(self):
        """output_path를 주면 loss 그래프 PNG 파일을 저장하는지 확인한다."""
        from train import plot_losses

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "loss.png"
            plot_losses([10, 20, 30], [0.5, 0.4, 0.3], [0.6, 0.5, 0.4], output_path=output_path)
            assert output_path.exists()


class TestTrainModel:
    """train_model의 확장 반환값과 체크포인트 저장을 확인한다."""

    def test_train_model_returns_metrics_and_saves_checkpoints(self):
        """train_model()이 train/val loss, tokens seen, sample text를 반환하고 best/last checkpoint를 저장하는지 확인한다."""
        from dataset import create_dataloader
        from model import GPTModel
        from train import train_model

        model = GPTModel(GPT_CONFIG_SMALL)
        train_loader = create_dataloader(
            list(range(300)),
            context_length=16,
            batch_size=4,
            shuffle=False,
        )
        val_loader = create_dataloader(
            list(range(300, 500)),
            context_length=16,
            batch_size=4,
            shuffle=False,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            train_losses, val_losses, tokens_seen, sample_texts = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                device=torch.device("cpu"),
                num_epochs=1,
                eval_freq=1,
                eval_iter=1,
                start_context="테스트",
                tokenizer=TinyTokenizer(),
                ckpt_freq=10,
                checkpoint_dir=checkpoint_dir,
            )

            assert len(train_losses) >= 1
            assert len(train_losses) == len(val_losses) == len(tokens_seen)
            assert sample_texts and "sample_text" in sample_texts[0]
            assert (checkpoint_dir / "best.pt").exists()
            assert (checkpoint_dir / "last.pt").exists()

    def test_train_model_recovers_from_sample_decode_error(self):
        """epoch 말미 샘플 decode가 깨져도 replacement decode로 학습 루프를 마무리해야 한다."""
        from dataset import create_dataloader
        from model import GPTModel
        from train import train_model

        model = GPTModel(GPT_CONFIG_SMALL)
        train_loader = create_dataloader(
            list(range(300)),
            context_length=16,
            batch_size=4,
            shuffle=False,
        )
        val_loader = create_dataloader(
            list(range(300, 500)),
            context_length=16,
            batch_size=4,
            shuffle=False,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            _, _, _, sample_texts = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                device=torch.device("cpu"),
                num_epochs=1,
                eval_freq=1,
                eval_iter=1,
                start_context="테스트",
                tokenizer=RecoveringTokenizer(),
                ckpt_freq=10,
                checkpoint_dir=checkpoint_dir,
            )

            assert sample_texts[0]["sample_text"] == "decoded with replacement"
