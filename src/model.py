# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))     # 정규화한 값을 다시 얼마나 키울지 학습하는 값. 처음엔 1이라 크기를 안 바꾼다.
        self.beta = nn.Parameter(torch.zeros(normalized_shape))     # 정규화한 값을 다시 얼마나 옮길지 학습하는 값. 처음엔 0이라 위치를 안 바꾼다.
        self.eps = eps                                              # 분산이 0에 가까울 때 나누기 오류가 나지 않게 더해주는 작은 값.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        mean = x.mean(dim=-1, keepdim=True)                         # 토큰 벡터 하나 안에서 평균을 구한다. batch나 seq 방향은 섞지 않는다.
        var = x.var(dim=-1, keepdim=True, unbiased=False)            # 같은 토큰 벡터 안에서 값들이 얼마나 퍼져 있는지 구한다.
        norm_x = (x - mean) / torch.sqrt(var + self.eps)             # 평균은 0 근처, 분산은 1 근처가 되게 맞춰 계산을 안정시킨다.
        return self.gamma * norm_x + self.beta                       # 정규화만 하면 표현력이 줄 수 있어서, 모델이 다시 크기와 위치를 조절하게 한다.


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """tanh 근사식으로 GELU 활성화 함수를 계산합니다."""
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )                                                           # ReLU처럼 딱 자르지 않고, 작은 값도 부드럽게 남겨서 학습 흐름을 덜 끊는다.


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # d_model -> mult*d_model -> d_model 구조의 작은 MLP.
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),                     # 각 토큰 벡터를 더 넓은 공간으로 펼쳐서 특징을 만들 여유를 준다.
            GELU(),                                                 # Linear만 있으면 결국 선형 계산이라, 중간에 비선형 변화를 넣는다.
            nn.Linear(mult * d_model, d_model),                     # 다음 블록과 shortcut 덧셈에 맞게 다시 d_model 크기로 돌려놓는다.
            nn.Dropout(dropout),                                    # 학습 때 일부 값을 꺼서 특정 계산에만 과하게 기대지 않게 한다.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FeedForward 네트워크를 통과시킵니다."""
        return self.net(x)                                          # Sequential에 넣어둔 Linear -> GELU -> Linear -> Dropout을 순서대로 실행한다.


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        # GPT 블록 안에서 쓰는 attention, ffn, layernorm, dropout.
        self.attention = MultiHeadAttention(
            d_model=d_model,                                        # 입력 token vector 길이를 그대로 attention 내부 차원으로 쓴다.
            n_heads=n_heads,                                        # 여러 head로 나눠서 서로 다른 관점의 문맥을 보게 한다.
            drop_rate=drop_rate,                                    # attention weight나 출력 일부를 학습 중에 랜덤으로 꺼준다.
            qkv_bias=qkv_bias,                                      # Q/K/V Linear에 bias를 둘지 정한다. GPT-2 설정은 보통 False다.
        )
        self.ffn = FeedForward(d_model=d_model, dropout=drop_rate)  # attention이 섞은 문맥 벡터를 토큰별로 한 번 더 가공한다.
        self.norm1 = LayerNorm(d_model)                             # attention 전에 입력 스케일을 정리하는 LayerNorm.
        self.norm2 = LayerNorm(d_model)                             # FFN 전에 다시 입력 스케일을 정리하는 LayerNorm.
        self.drop_shortcut = nn.Dropout(drop_rate)                  # residual로 더하기 전 출력 일부를 꺼서 과적합을 줄인다.

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """attention과 ffn을 residual connection으로 연결합니다."""
        shortcut = x                                               # attention을 지나기 전 원본을 잠깐 보관한다.
        x = self.norm1(x)                                          # attention 전에 값의 스케일을 한번 정리한다.
        x = self.attention(x, causal_mask=causal_mask)              # 토큰들이 이전 토큰 문맥을 참고해서 새 벡터를 만든다.
        x = self.drop_shortcut(x)                                  # attention 결과 일부를 학습 중 랜덤으로 꺼준다.
        x = x + shortcut                                           # attention 결과에 원본 입력을 더해 정보와 gradient가 잘 흐르게 한다.

        shortcut = x                                               # FFN을 지나기 전 현재 값을 다시 보관한다.
        x = self.norm2(x)                                          # FFN에 넣기 전에 다시 안정화한다.
        x = self.ffn(x)                                            # 각 토큰 벡터를 개별적으로 더 깊게 가공한다.
        x = self.drop_shortcut(x)                                  # FFN 결과에도 dropout을 적용한다.
        x = x + shortcut                                           # FFN 결과와 원본을 더해 깊은 모델에서도 정보가 끊기지 않게 한다.
        return x                                                   # shape은 처음과 같은 (batch, seq_len, d_model)로 유지된다.


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config                                       # 모델 설정값을 나중에 확인할 수 있게 객체에 저장한다.
        # embedding부터 lm_head까지 GPT 모델의 큰 흐름.
        self.embedding = InputEmbedding(
            vocab_size=config["vocab_size"],                       # 출력으로 맞춰야 할 전체 token 종류 수.
            emb_dim=config["emb_dim"],                             # token 하나를 표현할 벡터 길이, 즉 d_model.
            context_length=config["context_length"],               # 위치 임베딩이 준비할 수 있는 최대 문맥 길이.
            drop_rate=config["drop_rate"],                         # embedding 결과에도 dropout을 적용한다.
        )
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=config["emb_dim"],                     # 모든 블록이 같은 token vector 길이를 유지한다.
                    n_heads=config["n_heads"],                     # attention head 개수.
                    drop_rate=config["drop_rate"],                 # block 내부 dropout 비율.
                    qkv_bias=config["qkv_bias"],        # config에 없으면 GPT-2 기본처럼 False로 둔다.
                )
                for _ in range(config["n_layers"])                 # 같은 구조의 TransformerBlock을 n_layers개 쌓는다.
            ]
        )
        self.final_norm = LayerNorm(config["emb_dim"])             # 마지막 출력 전에 한 번 더 정규화해서 logits 계산을 안정시킨다.
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)  # 각 위치의 벡터를 vocab 전체 점수로 바꾼다.

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """
        x = self.embedding(idx)                                    # token id를 GPT가 계산할 수 있는 벡터로 바꾼다.
        x = self.blocks(x)                                         # 여러 transformer block을 지나며 문맥 정보를 쌓는다.
        x = self.final_norm(x)                                     # 출력 head에 넣기 전에 마지막으로 값을 정리한다.
        logits = self.lm_head(x)                                   # 각 위치마다 vocab 전체에 대한 다음 토큰 점수를 만든다.

        if targets is None:
            return logits                                          # 추론이나 생성에서는 정답이 없으니 점수만 반환한다.

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),                   # (B, T, vocab)을 한 줄짜리 문제들로 펼친다.
            targets.reshape(-1),                                   # 정답 token id도 같은 순서로 펼친다.
        )
        return loss, logits                                        # 학습에서는 loss로 업데이트하고, logits는 확인용으로 같이 돌려준다.


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]                          # 모델이 볼 수 있는 최대 길이만 남긴다.
        with torch.no_grad():                                      # 생성할 때는 학습용 gradient가 필요 없다.
            logits = model(idx_cond)                               # 현재까지의 토큰을 넣고 다음 토큰 후보 점수를 얻는다.
        logits = logits[:, -1, :]                                  # 마지막 토큰 위치의 다음 토큰 점수만 사용한다.
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)       # 가장 점수가 높은 토큰을 고른다.
        idx = torch.cat((idx, idx_next), dim=1)                     # 고른 토큰을 뒤에 붙이고 다음 반복에서 다시 입력으로 쓴다.

    return idx                                                      # 처음 입력 토큰 뒤에 새 토큰들이 이어 붙은 결과를 반환한다.
