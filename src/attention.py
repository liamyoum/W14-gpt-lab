# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
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
        self.d_model = d_model                  # 전체 임베딩 벡터 차원 (예: 768)
        self.n_heads = n_heads                  # attention head 개수 (예: 12)
        self.head_dim = d_model // n_heads      # head 하나당 차원 (예: 768/12 = 64)
        # TODO: qkv projection, output projection, dropout을 정의하세요.
        # 가중치 초기화
        self.W_query = nn.Linear(self.d_model, self.d_model, bias=qkv_bias)
        self.W_key = nn.Linear(self.d_model, self.d_model, bias=qkv_bias)
        self.W_value = nn.Linear(self.d_model, self.d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(self.d_model, self.d_model)  # head를 합친 최종 출력
        self.dropout = nn.Dropout(drop_rate)                    # attention weight에 적용할 dropout
        # causal mask를 모델 buffer로 등록해 device 이동 시 함께 다루고, forward에서 필요 크기로 갱신한다.
        self.register_buffer("mask", torch.empty(0, 0, dtype=torch.bool))

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        batch_size, seq_len, _ = x.shape  # x: (B, T, C)

        # qkv 벡터 계산
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        # 각 head가 독립적으로 attention을 계산할 수 있게 (B, T, C) -> (B, n_heads, T, head_dim)
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # 각 query 토큰이 각 key 토큰과 얼마나 관련 있는지 점수를 계산한다.
        attn_score = query @ key.transpose(-2, -1)
        # 점수 크기가 너무 커지지 않도록 head_dim의 제곱근으로 나눈다.
        attn_score = attn_score / (self.head_dim ** 0.5)

        if causal_mask:
            # 현재 길이보다 buffer가 작거나 device가 다르면 필요한 크기의 mask로 갱신한다.
            if self.mask.size(0) < seq_len or self.mask.device != x.device:
                self.mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1,
                )
            attn_score = attn_score.masked_fill(self.mask[:seq_len, :seq_len], float("-inf"))

        # 마지막 차원(T) 기준으로 softmax를 적용해 각 query의 attention 분포를 만든다.
        attn_weights = torch.softmax(attn_score, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # attention weight로 value를 가중합해 문맥 벡터를 만든다.
        context_vector = attn_weights @ value
        # head를 다시 합쳐 (B, T, C)로 되돌린 뒤 최종 projection을 적용한다.
        context_vector = context_vector.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        context_vector = self.out_proj(context_vector)
        
        return context_vector
