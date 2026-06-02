# -*- coding: utf-8 -*-
"""토큰 임베딩 + 위치 임베딩 과제 템플릿."""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    token ID를 Transformer 입력 벡터로 바꿉니다.

    구현할 구조:
    - token embedding: nn.Embedding(vocab_size, emb_dim)
    - position embedding: nn.Embedding(context_length, emb_dim)
    - token embedding + position embedding
    - dropout
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_length: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.context_length = context_length
        # 각 토큰 ID를 emb_dim 차원의 학습 가능한 벡터로 바꾼다.
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        # 각 위치(0, 1, 2, ...)에 대한 위치 정보를 벡터로 학습한다.
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        # 임베딩 단계에서도 dropout을 적용해 과적합을 줄인다.
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        batch_size, seq_len = x.shape

        # 토큰 ID -> 토큰 임베딩: (B, T) -> (B, T, C)
        token_embeds = self.token_embedding(x)

        # 위치 인덱스는 배치마다 같으므로 0 ~ seq_len-1만 만들면 된다.
        positions = torch.arange(seq_len, device=x.device)
        # 위치 임베딩: (T,) -> (T, C)
        position_embeds = self.position_embedding(positions)

        # broadcasting으로 (T, C)가 각 배치에 자동으로 더해진다.
        x = token_embeds + position_embeds
        x = self.dropout(x)
        return x
