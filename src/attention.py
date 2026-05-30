# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    GPT 방식 causal self-attention.

    흐름:
    x -> Q/K/V -> head 분리 -> QK^T -> mask -> softmax -> V 섞기
      -> head 합치기 -> output projection
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 같은 x를 query, key, value 세 관점으로 바꾼다.
        self.W_query = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_key = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, d_model, bias=qkv_bias)

        # head를 붙인 뒤 한 번 더 섞어준다.
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 토큰을 가린다.
            return_attention_weights: True이면 attention weight도 같이 반환한다.
        """
        B, T, C = x.shape

        # Q/K/V 생성. shape는 아직 (B, T, C)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # C를 head 개수와 head별 길이로 쪼갠다. (B, T, C) -> (B, T, H, D)
        queries = queries.view(B, T, self.n_heads, self.head_dim)
        keys = keys.view(B, T, self.n_heads, self.head_dim)
        values = values.view(B, T, self.n_heads, self.head_dim)

        # head별로 계산하기 좋게 head축을 앞으로 옮긴다. (B, T, H, D) -> (B, H, T, D)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 각 head에서 query token과 key token의 점수표를 만든다.
        # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        if causal_mask:
            # True인 칸은 미래 토큰이라 막을 위치다.
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        # 마지막 축(key token 방향)으로 softmax한다.
        # torch가 모든 batch/head/query 행에 대해 한 번에 처리한다.
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # attention weight 비율대로 value를 섞는다.
        # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        context_vec = attn_weights @ values

        # 다시 token축을 앞으로 돌려 head를 붙일 준비를 한다.
        context_vec = context_vec.transpose(1, 2)

        # contiguous는 transpose 뒤 메모리를 정리하고, view는 H*D를 다시 C로 붙인다.
        context_vec = context_vec.contiguous().view(B, T, self.d_model)

        context_vec = self.out_proj(context_vec)

        if return_attention_weights:
            return context_vec, attn_weights

        return context_vec
