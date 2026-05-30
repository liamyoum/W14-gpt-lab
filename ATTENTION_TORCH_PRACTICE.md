# attention.py 전에 해보는 torch 문법

이 문서는 "연습 파일을 어떻게 쓰는지"보다 `attention.py` 구현에 필요한 **torch 문법을 터미널에서 어떻게 직접 쳐보는지**에 맞춰 정리했습니다.

## 1. 시작

프로젝트 루트에서 Python REPL을 엽니다.

```bash
cd /Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab
conda run -n gpt-lab python
```

REPL 안에서 이 한 줄부터 실행합니다.

```python
from practice_attention_torch import demo, build_demo, split_heads, merge_heads
```

현재 기본 `python3`는 `3.13`이라 `torch` import가 바로 안 될 수 있습니다. 이 과제는 `gpt-lab` conda 환경의 Python `3.11`로 실행하는 쪽이 안전합니다.

## 2. 이 파일의 역할

`practice_attention_torch.py`는 아래 값을 미리 만들어 둡니다.

- `demo["x"]`: attention 입력
- `demo["qkv_proj"]`: `nn.Linear(d_model, 3 * d_model)`
- `demo["q"]`, `demo["k"]`, `demo["v"]`: projection 후 나눈 결과
- `demo["scores"]`: `q @ k^T`
- `demo["weights"]`: mask + softmax까지 적용한 결과
- `demo["context"]`: `weights @ v`

즉, "내가 지금 배우려는 문법"을 바로 실행해볼 재료를 넣어 둔 셈입니다.

## 3. attention.py에 필요한 문법만 연습하기

아래는 `forward()`에서 실제로 쓰게 될 문법입니다.

### 3-1. `nn.Linear`를 함수처럼 호출하기

무엇을 익히나:
`nn.Linear(...)`로 레이어를 만든 뒤 `layer(x)`처럼 호출하는 문법

직접 칠 것:

```python
demo["qkv_proj"]
qkv = demo["qkv_proj"](demo["x"])
qkv
```

무슨 뜻인가:
`demo["qkv_proj"]`는 projection 레이어이고, `demo["qkv_proj"](demo["x"])`는 입력 `x`를 그 레이어에 통과시킨다는 뜻입니다.

### 3-2. `chunk(3, dim=-1)`로 Q, K, V 나누기

무엇을 익히나:
마지막 차원을 3등분해서 여러 텐서로 받는 문법

직접 칠 것:

```python
q, k, v = demo["qkv"].chunk(3, dim=-1)
q
k
v
```

무슨 뜻인가:
`qkv`는 마지막 차원에 Q, K, V가 붙어 있는 상태라서 `chunk(3, dim=-1)`로 셋으로 나눕니다.

### 3-3. `view(...).transpose(...)`를 이어서 쓰기

무엇을 익히나:
head 차원을 끼워 넣고, 순서를 바꾸는 문법

직접 칠 것:

```python
q = demo["q"]
q.view(demo["batch_size"], demo["seq_len"], demo["n_heads"], demo["head_dim"])
q.view(demo["batch_size"], demo["seq_len"], demo["n_heads"], demo["head_dim"]).transpose(1, 2)
```

무슨 뜻인가:
첫 줄은 head 차원을 새로 끼워 넣는 것이고, 둘째 줄은 `(B, T, n_heads, head_dim)`을 `(B, n_heads, T, head_dim)` 순서로 바꾸는 것입니다.

같은 동작을 함수로 보면:

```python
split_heads(demo["q"], demo["n_heads"])
```

### 3-4. `transpose(-2, -1)`로 마지막 두 축 뒤집기

무엇을 익히나:
행렬곱 전에 `K`의 마지막 두 축을 바꾸는 문법

직접 칠 것:

```python
k_heads = split_heads(demo["k"], demo["n_heads"])
k_heads.transpose(-2, -1)
```

무슨 뜻인가:
attention score를 만들 때 `q @ k^T`가 필요하므로 마지막 두 차원을 뒤집습니다.

### 3-5. `@` 연산자로 행렬곱하기

무엇을 익히나:
PyTorch에서 `@`를 쓰면 `torch.matmul`처럼 동작하는 문법

직접 칠 것:

```python
q_heads = split_heads(demo["q"], demo["n_heads"])
k_heads = split_heads(demo["k"], demo["n_heads"])
q_heads @ k_heads.transpose(-2, -1)
```

같은 뜻을 다른 문법으로 쓰면:

```python
torch.matmul(q_heads, k_heads.transpose(-2, -1))
```

### 3-6. `/ math.sqrt(head_dim)`로 scaling 하기

무엇을 익히나:
텐서 전체를 스칼라 값으로 나누는 문법

직접 칠 것:

```python
import math
scores = demo["scores"]
scores / math.sqrt(demo["head_dim"])
```

무슨 뜻인가:
attention score가 너무 커지지 않게 `sqrt(head_dim)`으로 나눕니다.

### 3-7. `masked_fill(mask, float("-inf"))` 쓰기

무엇을 익히나:
특정 위치만 골라서 아주 작은 값으로 바꾸는 문법

직접 칠 것:

```python
scores = demo["scaled_scores"]
mask = demo["mask"]
scores.masked_fill(mask, float("-inf"))
```

무슨 뜻인가:
`mask == True`인 자리, 즉 미래 토큰 위치를 `-inf`로 바꿔서 softmax 후 확률이 0이 되게 만듭니다.

### 3-8. `torch.softmax(..., dim=-1)` 쓰기

무엇을 익히나:
마지막 차원을 확률 분포로 바꾸는 문법

직접 칠 것:

```python
masked_scores = demo["masked_scores"]
torch.softmax(masked_scores, dim=-1)
```

무슨 뜻인가:
각 query 위치가 "어느 key를 얼마나 볼지" 확률로 바뀝니다.

### 3-9. `weights @ v`로 context 만들기

무엇을 익히나:
attention weight와 value를 곱해서 최종 문맥 벡터를 만드는 문법

직접 칠 것:

```python
weights = demo["weights"]
v_heads = split_heads(demo["v"], demo["n_heads"])
weights @ v_heads
```

무슨 뜻인가:
확률 분포(`weights`)로 `v`를 가중합하는 단계입니다.

### 3-10. `transpose(...).contiguous().view(...)`로 head 합치기

무엇을 익히나:
차원 순서를 원래대로 돌린 뒤 메모리 배치를 정리하고 다시 펼치는 문법

직접 칠 것:

```python
context = demo["context"]
context.transpose(1, 2)
context.transpose(1, 2).contiguous()
context.transpose(1, 2).contiguous().view(
    demo["batch_size"],
    demo["seq_len"],
    demo["d_model"],
)
```

같은 동작을 함수로 보면:

```python
merge_heads(demo["context"])
```

무슨 뜻인가:
head별로 나뉜 결과를 다시 `(B, T, C)`로 합치는 단계입니다.

## 4. `forward()`와 1:1로 맞춰 보기

아래는 `attention.py`에서 거의 그대로 쓰게 될 형태입니다.

```python
qkv = self.qkv_proj(x)
q, k, v = qkv.chunk(3, dim=-1)

q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(self.head_dim)
scores = scores.masked_fill(mask, float("-inf"))

weights = torch.softmax(scores, dim=-1)
context = weights @ v
context = context.transpose(1, 2).contiguous().view(B, T, C)
```

REPL에서는 아래 순서로 그대로 쳐보면 됩니다.

```python
qkv = demo["qkv_proj"](demo["x"])
q, k, v = qkv.chunk(3, dim=-1)
q = q.view(demo["batch_size"], demo["seq_len"], demo["n_heads"], demo["head_dim"]).transpose(1, 2)
k = k.view(demo["batch_size"], demo["seq_len"], demo["n_heads"], demo["head_dim"]).transpose(1, 2)
v = v.view(demo["batch_size"], demo["seq_len"], demo["n_heads"], demo["head_dim"]).transpose(1, 2)
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(demo["head_dim"])
scores = scores.masked_fill(demo["mask"], float("-inf"))
weights = torch.softmax(scores, dim=-1)
context = weights @ v
context = context.transpose(1, 2).contiguous().view(
    demo["batch_size"], demo["seq_len"], demo["d_model"]
)
```

## 5. 가장 추천하는 연습 방식

처음부터 `shape`만 보지 말고, 아래처럼 한 문법씩 "직접 타이핑"해보는 방식이 제일 좋습니다.

1. `qkv = demo["qkv_proj"](demo["x"])`
2. `q, k, v = qkv.chunk(3, dim=-1)`
3. `q = q.view(...).transpose(1, 2)`
4. `scores = q @ k.transpose(-2, -1)`
5. `scores = scores.masked_fill(demo["mask"], float("-inf"))`
6. `weights = torch.softmax(scores, dim=-1)`
7. `context = weights @ v`
8. `context = context.transpose(1, 2).contiguous().view(...)`

이 순서가 손에 익으면 `attention.py` 구현 자체는 거의 번역 작업처럼 느껴질 겁니다.
