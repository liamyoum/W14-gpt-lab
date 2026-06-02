# -*- coding: utf-8 -*-
"""GPT 사전 학습에 필요한 loss 계산, 생성, 체크포인트, 학습 루프."""

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
    """한 배치의 next-token prediction loss를 계산합니다."""
    input_batch = input_batch.to(device)  # 모델과 같은 장치에서 계산해야 하므로 input을 CPU/GPU로 옮긴다.
    target_batch = target_batch.to(device)  # 정답 target도 loss 계산에 쓰이므로 같은 장치로 맞춘다.

    loss, logits = model(input_batch, targets=target_batch)  # targets를 같이 넘기면 GPTModel이 loss까지 계산해준다.

    return loss  # 이 함수는 한 배치의 loss만 필요하므로 logits는 버리고 loss만 돌려준다.


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """DataLoader에서 몇 개 배치의 평균 loss를 계산합니다."""
    model.eval()  # 평균 loss 확인 때는 dropout을 꺼서 결과가 흔들리지 않게 한다.

    total_loss = 0.0  # 배치별 loss를 계속 더해둘 변수.

    if len(data_loader) == 0:
        return float("nan")  # 데이터가 없으면 평균을 낼 수 없으니 NaN으로 표시한다.

    if num_batches is None:
        num_batches = len(data_loader)  # 따로 제한이 없으면 loader 전체를 평가한다.
    else:
        num_batches = min(num_batches, len(data_loader))  # 요청한 배치 수가 실제 배치 수보다 크면 실제 개수까지만 본다.

    with torch.no_grad():  # 평가용 loss 계산에서는 weight를 고치지 않으므로 gradient 기록을 만들 필요가 없다.
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            # enumerate를 쓰면 현재 몇 번째 배치인지 batch_idx로 확인할 수 있다.
            if batch_idx >= num_batches:
                break  # 정해둔 개수만큼만 loss를 보고 멈춘다.

            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 한 배치의 loss를 계산한다.
            total_loss += loss.item()  # tensor loss를 Python 숫자로 바꿔 누적한다.

    return total_loss / num_batches  # 여러 배치 loss의 평균을 반환한다.


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """모델과 optimizer의 현재 학습 상태를 파일로 저장합니다."""
    checkpoint = {
        "model_state_dict": model.state_dict(),  # 모델의 모든 학습 파라미터(weight, bias 등)를 저장한다.
        "optimizer_state_dict": optimizer.state_dict(),  # AdamW 같은 optimizer의 내부 진행 상태도 같이 저장한다.
        "epoch": epoch,  # 몇 번째 epoch에서 저장했는지 기록한다.
        "global_step": global_step,  # 전체 batch 업데이트를 몇 번 했는지 기록한다.
    }

    torch.save(checkpoint, path)  # 위 정보를 하나의 .pt 파일로 저장한다.


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """저장된 checkpoint를 읽어 모델과 optimizer 상태를 복원합니다."""
    checkpoint = torch.load(path, map_location=device)  # 저장 당시 장치와 달라도 지금 device에 맞춰 불러온다.

    model.load_state_dict(checkpoint["model_state_dict"])  # 저장해둔 모델 파라미터를 현재 model에 덮어쓴다.

    if optimizer is not None:
        # 이어서 학습할 때는 optimizer 상태까지 복원해야 이전 학습 흐름을 자연스럽게 이어갈 수 있다.
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]  # 저장했던 epoch 번호를 꺼낸다.
    global_step = checkpoint["global_step"]  # 저장했던 전체 step 번호를 꺼낸다.

    return epoch, global_step  # 학습을 어디서 이어갈지 알려주기 위해 반환한다.


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """temperature와 top-k를 사용해 다음 토큰을 반복 생성합니다."""
    model.eval()  # 생성할 때는 학습이 아니므로 dropout을 꺼둔다.

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # context_length를 넘지 않게 마지막 토큰들만 모델에 넣는다.

        with torch.no_grad():
            logits = model(idx_cond)  # 현재까지의 토큰을 보고 다음 토큰 후보 점수를 만든다.

        if isinstance(logits, tuple):
            logits = logits[1]  # 혹시 (loss, logits)가 오면 생성에 필요한 logits만 꺼낸다.

        logits = logits[:, -1, :]  # 새로 붙일 토큰은 마지막 위치 다음이므로 마지막 위치 점수만 쓴다.

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)  # 점수가 높은 후보 top_k개만 뽑는다.
            min_top_logit = top_logits[:, -1].unsqueeze(-1)  # top_k 안에서 가장 낮은 점수를 기준선으로 둔다.
            logits = logits.masked_fill(logits < min_top_logit, -torch.inf)  # 기준선보다 낮은 후보는 확률 0으로 막는다.

        if temperature == 0.0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # temperature 0이면 랜덤 없이 최고 점수 토큰만 고른다.
        else:
            logits = logits / temperature  # 낮으면 보수적, 높으면 다양한 토큰이 뽑히기 쉬워진다.
            probs = torch.softmax(logits, dim=-1)  # vocab 점수(logits)를 확률로 바꾼다.
            idx_next = torch.multinomial(probs, num_samples=1)  # 확률 분포에 따라 다음 토큰 하나를 샘플링한다.

        idx = torch.cat((idx, idx_next), dim=1)  # 새로 뽑은 토큰을 기존 입력 뒤에 붙인다.

        if eos_id is not None and (idx_next == eos_id).all():
            break  # 모든 배치에서 문장 끝 토큰이 나오면 생성을 멈춘다.

    return idx  # 처음 입력 뒤에 생성 토큰들이 이어 붙은 결과.


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
    """시작 문장을 encode해서 생성한 뒤 다시 decode해 출력합니다."""
    model.eval()  # 샘플 확인은 학습이 아니므로 dropout을 끈다.

    encoded = tokenizer.encode(start_context)  # 사람이 읽는 문자열을 token id 리스트로 바꾼다.
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)  # 모델 입력에 맞게 batch 차원을 붙인다.

    out = generate(
        model=model,
        idx=idx,  # 시작 token id.
        max_new_tokens=max_new_tokens,  # 새로 생성할 token 개수.
        context_size=context_size,  # 모델이 한 번에 볼 수 있는 최대 token 길이.
        temperature=temperature,  # 생성 다양성 조절값.
        top_k=top_k,  # 후보를 상위 k개로 제한하는 옵션.
    )

    decoded_text = tokenizer.decode(out[0].tolist())  # batch 첫 번째 결과를 token id 리스트로 꺼내 문자열로 복원한다.
    print(decoded_text)  # 학습 중간에 모델이 어떤 문장을 만드는지 확인한다.

    model.train()  # 학습 루프 중 호출될 수 있으니 다시 train 모드로 돌려놓는다.


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
) -> list[float]:
    """GPT 모델을 여러 epoch 동안 학습하고 epoch별 평균 train loss를 반환합니다."""
    train_losses = []  # epoch가 끝날 때마다 평균 train loss를 저장한다.

    model.to(device)  # 모델 파라미터를 학습할 장치로 옮긴다.

    for epoch in range(start_epoch, num_epochs):
        model.train()  # 실제 학습 단계이므로 dropout을 켠다.
        epoch_loss = 0.0  # 현재 epoch의 loss 누적값.
        num_batches = 0  # 현재 epoch에서 처리한 batch 개수.

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 이전 batch의 gradient가 남아 있지 않게 매번 초기화한다.

            loss = calc_loss_batch(input_batch, target_batch, model, device)  # input -> logits -> target 비교 -> loss 계산.

            loss.backward()  # loss를 줄이려면 각 파라미터를 어느 방향으로 고칠지 계산한다.
            optimizer.step()  # 계산된 gradient를 이용해 실제 모델 파라미터를 업데이트한다.

            epoch_loss += loss.item()  # epoch 평균을 내기 위해 batch loss를 더한다.
            num_batches += 1  # 평균 계산에 쓸 batch 수를 센다.
            global_step += 1  # 전체 학습 step을 하나 증가시킨다.

            if eval_freq > 0 and global_step % eval_freq == 0:
                # 일정 step마다 train/val loss와 샘플 출력을 확인한다.
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

                print(
                    f"Epoch {epoch + 1}, step {global_step}: "
                    f"train loss {train_loss:.4f}, val loss {val_loss:.4f}"
                )

                generate_and_print_sample(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    start_context=start_context,
                    context_size=model.config["context_length"],  # 모델이 학습한 최대 문맥 길이에 맞춰 생성한다.
                )

                model.train()  # loss 평가/샘플 생성 후 다시 학습 모드로 돌아온다.

            if ckpt_freq is not None and ckpt_freq > 0 and global_step % ckpt_freq == 0:
                # 오래 걸리는 학습은 중간 저장을 해둬야 끊겨도 이어서 할 수 있다.
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    path=f"checkpoint_step_{global_step}.pt",
                )

        if num_batches > 0:
            train_losses.append(epoch_loss / num_batches)  # epoch 하나가 끝나면 평균 train loss를 기록한다.

    return train_losses  # plot_losses 등에서 볼 수 있게 loss 기록을 반환한다.


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 loss 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
