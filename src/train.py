# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch
from pathlib import Path

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """TODO: 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss, logits = model(input_batch, targets = target_batch)
    
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None, # 몇 개 batch만 평가할지 지정하는 값
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    if len(data_loader) == 0:
        return float("nan")
    
    if num_batches is None:
        num_batches = len(data_loader) # num_batches 지정되지 않으면 모든 배치 순회
    else:
        num_batches = min(num_batches, len(data_loader)) # num_batches가 데이터 로더에 있는 배치 개수보다 크면 배치 횟수를 데이터 로더에 있는 총 배치 개수로 지정

    if num_batches == 0:
        return float("nan")
    
    was_training = model.training
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
    
    if was_training:
        model.train()
    
    return total_loss / num_batches


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    torch.save({
        "model_state_dict": model.state_dict(), # 모델의 모든 학습 파라미터 값 (각종 weights)
        "optimizer_state_dict": optimizer.state_dict(), # optimizer의 내부 상태 (lr, step 수, 1차 & 2차 모멘트 추정값 등)
        "epoch": epoch, # 몇 번째 epoch까지 진행했는지
        "global_step": global_step # 몇 번의 batch update가 진행됐는지
        },
        path
    )


def load_checkpoint(
    model: GPTModel, # checkpoint 가중치를 넣을 GPT 모델 객체
	optimizer: torch.optim.Optimizer | None,#optimizer 상태를 복원할 객체. 없으면 `None` 가능
	path: str, # checkpoint 파일 경로
	device: torch.device, # checkpoint를 불러올 장치. CPU 또는 GPU
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (checkpoint["epoch"], checkpoint["global_step"])


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_logit = top_logits[:, -1, None]
            logits = torch.where(
                condition = logits < min_logit,
                input = torch.tensor(float("-inf")).to(logits.device),
                other = logits
            )
        
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim = -1, keepdim = True)
        
        if idx_next == eos_id:
            break
            
        idx = torch.cat((idx, idx_next), dim = 1)
    
    return idx



def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    was_training = model.training
    model.eval()
    encoded = tokenizer.encode(start_context)
    encoded = torch.tensor(encoded, dtype = torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        token_ids = generate(model, encoded, max_new_tokens, context_size, temperature, top_k)

    decoded_text = tokenizer.decode(token_ids[0].tolist())
    print(decoded_text.replace("\n", " "))
    if was_training:
        model.train()

def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> tuple[list[float], list[float]]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    # 손실과 지금까지 처리한 토큰 수를 추적하기 위해 리스트 초기화
    train_losses = []
    val_losses = []
    track_tokens_seen = []
    tokens_seen = 0

    # Main Training Loop
    for epoch in range(start_epoch, num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # 이전 배치 반복에서 얻은 손실 gradient 초기화
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # 손실 gradient 계산
            optimizer.step() # 손실 gradient 사용해 모델 weights update

            tokens_seen += input_batch.numel() # input_batch 내부 전체 원소 개수 반환
            global_step += 1

            # 추가적인 평가 단계
            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"에포크 {epoch+1} (Step {global_step:06d}): "
                      f"훈련 손실 {train_loss:.3f}, "
                      f"검증 손실 {val_loss:.3f}"
				)
        
        generate_and_print_sample(model, tokenizer, device, start_context)
        
        if ckpt_freq is not None and (epoch + 1) % ckpt_freq == 0:
            Path("checkpoints").mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, epoch + 1, global_step, f"checkpoints/ckpt_epoch_{epoch + 1}.pt")
        
    return (train_losses, val_losses)


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Evaluation step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
