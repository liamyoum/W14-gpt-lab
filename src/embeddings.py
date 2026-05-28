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
        super().__init__() # super().__init__()은 부모 클래스(torch.nn.Module)의 초기화 메서드를 실행하는 코드
        self.emb_dim = emb_dim
        self.context_length = context_length
        # TODO: token_embedding, position_embedding, dropout을 정의하세요.
        self.vocab_size = vocab_size
        self.drop_rate = drop_rate
        
        self.token_embedding_layer = torch.nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_embedding_layer = torch.nn.Embedding(self.context_length, self.emb_dim)
        self.dropout = torch.nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        # 여기 코드 복습 + 정리하기
        token_embeddings = self.token_embedding_layer(x)
        pos_embeddings = self.pos_embedding_layer(torch.arange(x.shape[1], device = x.device))

        input_embeddings = token_embeddings + pos_embeddings
        input_embeddings = self.dropout(input_embeddings)
        return input_embeddings
