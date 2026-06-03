# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

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
    """한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    
    return loss
    raise NotImplementedError("calc_loss_batch를 구현하세요.")


def calc_loss_loader(
    data_loader, # input과 target을 batch 단위로 꺼낸다.
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """data_loader의 평균 loss를 계산합니다."""
    total_loss = 0.
    
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else: 
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
        
    return total_loss / num_batches
        
    raise NotImplementedError("calc_loss_loader를 구현하세요.")


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    check_point = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, 
        "global_step": global_step,
    }
    torch.save(check_point, path)


def _normalize_checkpoint_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """다른 구현에서 저장한 GPT 체크포인트 키를 현재 모델 이름으로 맞춥니다."""
    if "tok_emb.weight" in state_dict:
        return state_dict

    direct_key_map = {
        "embedding.token_embedding_layer.weight": "tok_emb.weight",
        "embedding.pos_embedding_layer.weight": "pos_emb.weight",
        "final_layernorm.gamma": "final_norm.gamma",
        "final_layernorm.beta": "final_norm.beta",
        "lm_head.weight": "out_head.weight",
    }
    normalized_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = direct_key_map.get(key, key)
        if ".ffn." in new_key:
            new_key = new_key.replace(".ffn.", ".ff.")
        normalized_state_dict[new_key] = value
    return normalized_state_dict


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    check_point = torch.load(path, map_location = device)
    state_dict = _normalize_checkpoint_state_dict(check_point["model_state_dict"])
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    missing_keys = [key for key in missing_keys if not key.endswith(".att.mask")]
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "checkpoint와 현재 모델 구조가 맞지 않습니다. "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )
    
    if optimizer is not None:
        optimizer.load_state_dict(check_point["optimizer_state_dict"])

    epoch = check_point["epoch"]    
    global_step = check_point["global_step"]
    return epoch, global_step


def generate(
    model: GPTModel,
    idx: torch.Tensor, # 현재 생성된 token id들이 있는 tensor
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    for _ in range(max_new_tokens):
        context_tokens = idx[:, -context_size:] # 모든 batch에 대해 각 sequence의 뒤에서부터 context만 인덱싱
        with torch.no_grad():
            logits = model(context_tokens)
        logits = logits[:, -1, :]
        # top-k 필터링
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where( # logits가 min_val 보다 작으면 -inf로 바꾼다
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        # temperature 적용
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
        
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
) -> str:
    """학습 정도를 확인하기 위해 학습 도중 일정 step마다 생성 샘플을 출력한다."""
    model.eval()
    
    ids = tokenizer.encode(start_context) # 정수 id list
    # 현재까지 생성된 token ids sequence. generate에서 하나씩 늘어난다.
    idx = torch.tensor(ids, device=device).unsqueeze(0) # list을 (1, T) tensor로 변환, batch 차원 1추가.
    # 텍스트를 토큰 단위로 생성해서 idx(prompt) + 새로 생성된 tokens을 만든다.
    token_ids = generate(  
                        model,
                        idx,
                        max_new_tokens,
                        context_size,
                        temperature = temperature,
                        top_k = top_k,
                        eos_id = tokenizer.get_eos_id()
                        )
    # batch dim 제거 후 decode를 통해 tensor를 list로 변환한다.
    try:
        text = tokenizer.decode(token_ids[0].tolist())
    except UnicodeDecodeError:
        # Generated byte sequences are not guaranteed to form valid UTF-8 mid-training.
        text = tokenizer.decode(token_ids[0].tolist(), errors="replace")
    
    # 텍스트의 줄바꿈을 제거해 한 줄로 만든다.
    clean_text = text.replace("\n", " ")
    print(clean_text)
    
    model.train()
    return clean_text


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
    checkpoint_dir: str | Path | None = None,
    start_epoch: int = 0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    global_step: int = 0,
) -> tuple[list[float], list[float], list[int], list[dict[str, int | str]]]:
    """사전 학습 루프를 돌리고 train/val loss, tokens seen, 생성 샘플을 반환합니다."""
    # 손실, 처리한 토큰 리스트 초기화.
    train_losses, val_losses, track_tokens_seen = [], [], []
    sample_texts: list[dict[str, int | str]] = []
    token_seen = 0
    best_val_loss = float("inf")
    last_epoch = start_epoch - 1
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, num_epochs):
        last_epoch = epoch
        model.train()
        # 모델의 사전훈련을 시행한다.
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device) # cross entropy loss를 계산한다.
            loss.backward() # loss gradient를 계산한다.
            optimizer.step() # 모델 가중치를 업데이트한다.
            
            token_seen += input_batch.numel() # 생성한 토큰 수에 tensor 안의 원소 개수를 더한다.
            global_step += 1 # 지금까지 처리한 batch 수를 하나 늘린다.
        
        # eval_freq 마다 loss를 출력한다.
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                    val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                track_tokens_seen.append(token_seen)
                print(f"epoch {epoch+1} (step: {global_step:06d}): "
                    f"train loss {train_loss:.3f}, "
                    f"val loss {val_loss:.3f}"
                    )
                if checkpoint_dir is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, epoch, global_step, str(checkpoint_dir / "best.pt"))
                model.train()
            # 체크포인트를 저장한다.
            if ckpt_freq is not None and global_step % ckpt_freq == 0:
                ckpt_dir = checkpoint_dir if checkpoint_dir is not None else Path("checkpoints")
                ckpt_dir.mkdir(exist_ok=True)
                ckpt_path = ckpt_dir / f"checkpoint_step_{global_step}.pt"
                save_checkpoint(model, optimizer, epoch, global_step, str(ckpt_path))

        # 각 epoch 후 샘플 텍스트를 출력한다.
        sample_text = generate_and_print_sample(model, tokenizer, device, start_context)
        sample_texts.append(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "sample_text": sample_text,
            }
        )
    
    if checkpoint_dir is not None:
        save_checkpoint(model, optimizer, last_epoch, global_step, str(checkpoint_dir / "last.pt"))
        best_path = checkpoint_dir / "best.pt"
        if not best_path.exists():
            save_checkpoint(model, optimizer, last_epoch, global_step, str(best_path))

    return train_losses, val_losses, track_tokens_seen, sample_texts
            
    

def plot_losses(
    x_values_or_train_losses: list[float] | list[int],
    train_losses_or_val_losses: list[float] | None = None,
    val_losses: list[float] | None = None,
    output_path: str | Path | None = None,
) -> None:
    """훈련/검증 손실 그래프를 그리고, 필요하면 파일로 저장합니다."""
    if val_losses is None:
        x_values = list(range(1, len(x_values_or_train_losses) + 1))
        train_losses = x_values_or_train_losses
        val_losses = train_losses_or_val_losses
        xlabel = "Step"
    else:
        x_values = x_values_or_train_losses
        train_losses = train_losses_or_val_losses
        xlabel = "Tokens Seen"

    plt.figure()
    plt.plot(x_values, train_losses, label="Train")
    if val_losses is not None:
        plt.plot(x_values, val_losses, label="Val")
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return
    plt.show()
