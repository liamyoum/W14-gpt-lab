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
        # TODO: token_embedding, position_embedding, dropout을 정의하세요.
        self.token_embedding = nn.Embedding(vocab_size, self.emb_dim)
        self.position_embedding = nn.Embedding(context_length, self.emb_dim)
        self.dropout = nn.Dropout(drop_rate)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: token embedding과 position embedding을 더한 뒤 dropout을 적용합니다.

        Args:
            x: (batch_size, seq_len) token IDs

        Returns:
            (batch_size, seq_len, emb_dim)
        """
        
        batch_size, seq_len = x.shape                               # embedding 할때 받아와야되는 x의 shape의 batch_size, seq_len 중 seq_len 이 필요하다 seq_len은 이번 forward에 실제 들어온 token 개수다.
                                                                    # 학습 batch에서는 보통 context_length와 같지만, 짧은 입력에서는 context_length보다 작을 수 있다.
        token_vector = self.token_embedding(x)                      # token_vector는 x 토큰에 대한 전체 embedding table을  token_embedding layer를 통해 벡터를 추가해준다

        pos_id = torch.arange(seq_len, device = x.device)           # pos_id는 위에 설명했다시피 batch_size의 행의 열인 seq_len을 받아와서 pytorch로 arange로 seq_len까지 길이를 재고 그 길이만큼 id를 붙혀준다 
                                                                    # 그리고 device부분은 gpu 와 cpu 나눠서 변수가 들어갈수있어서 그걸 통일시킨거다
        
        pos_vector = self.position_embedding(pos_id)                # 위치 번호들을 position_embedding table에서 찾아 각 위치의 emb_dim 길이 vector로 바꾼다.
        embeddings = token_vector + pos_vector                      # embeddings에 token_vector와 pos_vector를 넣어줌으로써 각 token vector에 해당 위치 vector를 더해서, token 의미 정보와 위치 정보를 함께 담은 입력 embedding을 만든다.
        embeddings = self.dropout(embeddings)                       # 학습 중 embeddings의 일부 값을 랜덤하게 0으로 만들어 특정 값에 과하게 의존하지 않게 한다.
        return embeddings                                           # 이번 입력 x에 대한 embedding 결과 tensor
