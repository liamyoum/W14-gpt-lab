# -*- coding: utf-8 -*-
"""attention.py 구현 전에 터미널에서 torch 문법을 연습하는 스크립트.

실행 예시:
    conda run -n gpt-lab python practice_attention_torch.py
    conda run -n gpt-lab python -i practice_attention_torch.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

torch.set_printoptions(precision=3, sci_mode=False)


def show_tensor(name: str, tensor: torch.Tensor) -> None:
    """텐서 이름, shape, 값을 함께 출력한다."""
    print(f"\n[{name}]")
    print(f"shape={tuple(tensor.shape)}")
    print(tensor)


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, T, C) -> (B, n_heads, T, head_dim)"""
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // n_heads
    return x.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, n_heads, T, head_dim) -> (B, T, C)"""
    batch_size, n_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, n_heads * head_dim)


def make_causal_mask(seq_len: int) -> torch.Tensor:
    """상삼각(True) mask를 만든다. True 위치는 미래 토큰이다."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)


def build_demo(
    batch_size: int = 2,
    seq_len: int = 4,
    d_model: int = 8,
    n_heads: int = 2,
    seed: int = 7,
) -> dict[str, object]:
    """attention.py에서 쓰게 될 중간 결과를 모두 담아 반환한다."""
    torch.manual_seed(seed)
    head_dim = d_model // n_heads

    x = torch.randn(batch_size, seq_len, d_model)
    qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
    out_proj = nn.Linear(d_model, d_model, bias=False)

    qkv = qkv_proj(x)
    q, k, v = qkv.chunk(3, dim=-1)

    q_heads = split_heads(q, n_heads)
    k_heads = split_heads(k, n_heads)
    v_heads = split_heads(v, n_heads)

    scores = q_heads @ k_heads.transpose(-2, -1)
    scaled_scores = scores / math.sqrt(head_dim)

    mask = make_causal_mask(seq_len)
    masked_scores = scaled_scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)

    context = weights @ v_heads
    merged = merge_heads(context)
    output = out_proj(merged)

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "x": x,
        "qkv_proj": qkv_proj,
        "out_proj": out_proj,
        "qkv": qkv,
        "q": q,
        "k": k,
        "v": v,
        "q_heads": q_heads,
        "k_heads": k_heads,
        "v_heads": v_heads,
        "scores": scores,
        "scaled_scores": scaled_scores,
        "mask": mask,
        "masked_scores": masked_scores,
        "weights": weights,
        "context": context,
        "merged": merged,
        "output": output,
    }


def print_demo(demo: dict[str, object]) -> None:
    """연습에 필요한 핵심 텐서만 순서대로 보여준다."""
    show_tensor("1. input x", demo["x"])
    show_tensor("2. qkv projection output", demo["qkv"])
    show_tensor("3. q", demo["q"])
    show_tensor("4. k", demo["k"])
    show_tensor("5. v", demo["v"])
    show_tensor("6. q split into heads", demo["q_heads"])
    show_tensor("7. k split into heads", demo["k_heads"])
    show_tensor("8. attention scores = q @ k^T", demo["scores"])
    show_tensor("9. scaled scores", demo["scaled_scores"])
    show_tensor("10. causal mask", demo["mask"])
    show_tensor("11. masked scores", demo["masked_scores"])
    show_tensor("12. attention weights", demo["weights"])
    show_tensor("13. context per head", demo["context"])
    show_tensor("14. merged heads", demo["merged"])
    show_tensor("15. final output", demo["output"])

    row_sums = demo["weights"][0, 0].sum(dim=-1)
    future_weights = demo["weights"][0, 0].triu(diagonal=1)
    show_tensor("16. first head row sums", row_sums)
    show_tensor("17. future attention weights should be 0", future_weights)


def print_repl_tips() -> None:
    """python -i 실행 시 바로 따라 해볼 명령을 안내한다."""
    print("\n[REPL syntax practice]")
    print("1. qkv = demo['qkv_proj'](demo['x'])")
    print("   nn.Linear 레이어를 함수처럼 호출하는 문법")
    print("2. q, k, v = qkv.chunk(3, dim=-1)")
    print("   마지막 차원을 3등분해서 q, k, v로 받는 문법")
    print("3. q = q.view(demo['batch_size'], demo['seq_len'], demo['n_heads'], demo['head_dim']).transpose(1, 2)")
    print("   head 차원을 끼워 넣고 순서를 바꾸는 문법")
    print("4. scores = q @ k.transpose(-2, -1)")
    print("   마지막 두 축을 뒤집은 뒤 행렬곱하는 문법")
    print("5. scores = scores / math.sqrt(demo['head_dim'])")
    print("   score scaling 문법")
    print("6. scores = scores.masked_fill(demo['mask'], float('-inf'))")
    print("   mask가 True인 자리만 -inf로 바꾸는 문법")
    print("7. weights = torch.softmax(scores, dim=-1)")
    print("   마지막 차원을 확률로 바꾸는 문법")
    print("8. context = weights @ v")
    print("   attention weight와 value를 곱하는 문법")
    print("9. context = context.transpose(1, 2).contiguous().view(demo['batch_size'], demo['seq_len'], demo['d_model'])")
    print("   head를 다시 합치는 문법")


def main() -> dict[str, object]:
    """데모를 만들고 출력한다."""
    current_demo = build_demo()

    print("attention.py 구현 전에 익혀둘 torch 문법")
    print("- nn.Linear")
    print("- chunk(dim=-1)")
    print("- view / transpose / contiguous")
    print("- @ 또는 torch.matmul")
    print("- masked_fill")
    print("- torch.softmax(dim=-1)")

    print_demo(current_demo)
    print_repl_tips()
    return current_demo


# import 후에도 바로 만져볼 수 있게 기본 demo를 준비해 둔다.
demo = build_demo()


if __name__ == "__main__":
    demo = main()
