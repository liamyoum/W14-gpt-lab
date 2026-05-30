# Multi-Head Self-Attention 학습 정리

`src/attention.py` 파일을 처음부터 끝까지 읽으면서 정리한 내용입니다.

---

## 목차

1. [PyTorch 기초](#1-pytorch-기초)
2. [Q/K/V Projection](#2-qkv-projection)
3. [가중치 초기화 블록](#3-가중치-초기화-블록)
4. [Dropout과 Causal Mask 준비](#4-dropout과-causal-mask-준비)
5. [Python/PyTorch 문법 다지기](#5-pythonpytorch-문법-다지기)
6. [forward 메서드 시그니처](#6-forward-메서드-시그니처)
7. [입력 모양 분해와 Q/K/V 계산](#7-입력-모양-분해와-qkv-계산)
8. [선형 변환의 동작 원리 (심화)](#8-선형-변환의-동작-원리-심화)
9. [Head 분리: view + transpose](#9-head-분리-view--transpose)
10. [PyTorch 행렬곱 규칙](#10-pytorch-행렬곱-규칙)
11. [Attention Score 계산](#11-attention-score-계산)
12. [Causal Mask 적용](#12-causal-mask-적용)
13. [Softmax와 Dropout](#13-softmax와-dropout)
14. [출력 생성](#14-출력-생성)
15. [메모리 심화: view vs reshape, 연속/불연속](#15-메모리-심화-view-vs-reshape-연속불연속)
16. [전체 forward 요약](#16-전체-forward-요약)

---

## 1. PyTorch 기초

### 1-1. `torch.nn`이 하는 일

**한 문장으로:** 신경망을 구성하는 레이어(선형 변환, 드롭아웃 등)와 학습 가능한 파라미터를 편리하게 관리해주는 PyTorch의 핵심 모듈.

`nn`은 "Neural Network"의 약자. 딥러닝 모델을 만들 때 필요한 재료 박스 같은 것.

| `nn`이 제공하는 것 | 이 파일에서 쓰인 예 |
|---|---|
| 학습 가능한 레이어 | `nn.Linear` (행렬 곱) |
| 정규화 기법 | `nn.Dropout` |
| 모든 레이어의 기반 클래스 | `nn.Module` |

`nn.Module`을 상속받으면 파라미터 자동 추적, GPU 이동(`to(device)`), 저장/불러오기 같은 기능을 공짜로 얻음.

### 1-2. 선형 변환이란?

입력에 **가중치를 곱하고 편향을 더하는** 연산.

수식: `y = Wx + b`

**직관:** 입력 벡터를 다른 차원의 벡터로 "변환"하는 것. 예를 들어 토큰 임베딩 벡터(768차원) → Query 벡터(768차원).

**"선형"인 이유:** 곱하기와 더하기만 쓰기 때문에 그래프가 직선/평면 형태. 비선형성은 별도로 활성화 함수가 담당.

---

## 2. Q/K/V Projection

### 2-1. 개념

같은 입력 벡터를 세 가지 **역할**로 분리해서 변환하는 것.

- **Q (Query):** "나는 어떤 정보를 찾고 있나?" (질문하는 토큰)
- **K (Key):** "나는 어떤 정보를 갖고 있나?" (검색 대상 토큰)
- **V (Value):** "실제로 전달할 내용은 뭔가?" (전달되는 정보)

같은 입력 벡터를 세 개의 서로 다른 선형 변환으로 통과시켜 Q, K, V를 각각 만들어낸다.

### 2-2. 왜 "투영(Projection)"이라고 부르나

수학에서 투영은 **고차원 공간의 벡터를 특정 방향으로 눌러서 다른 공간에 표현**하는 것.

예: 3D 물체를 벽에 그림자로 투영하면 2D가 된다. 마찬가지로:
- 768차원 임베딩을 → "Query 관점에서 본 공간"으로 투영
- 768차원 임베딩을 → "Key 관점에서 본 공간"으로 투영

같은 원본 벡터인데, **어떤 가중치 행렬로 투영하느냐**에 따라 완전히 다른 의미를 가진 벡터가 된다.

→ projection = "이 공간에서 저 공간으로 선형 변환"이라는 수학 용어를 그대로 가져온 것.

---

## 3. 가중치 초기화 블록

```python
self.W_query = nn.Linear(self.d_model, self.d_model, bias=qkv_bias)
self.W_key = nn.Linear(self.d_model, self.d_model, bias=qkv_bias)
self.W_value = nn.Linear(self.d_model, self.d_model, bias=qkv_bias)
self.out_proj = nn.Linear(self.d_model, self.d_model)
```

### 3-1. 중요한 오해 짚기

Q/K/V 세 개의 `Linear`는 **순차적이 아니라 병렬**. 같은 입력 `x`가 세 개 레이어로 **동시에** 들어가서 각각 Q, K, V를 만들어낸다.

```
        ┌─ W_query ─→ Q
   x ───┼─ W_key   ─→ K
        └─ W_value ─→ V
```

### 3-2. `nn.Linear(in, out)`가 실제로 만드는 것

`nn.Linear(768, 768)`을 호출하면 내부에 두 개의 텐서가 생성됨.

- **가중치 행렬 W**: `(768, 768)` 모양
- **편향 벡터 b**: `(768,)` 모양 (단, 이 코드는 `bias=qkv_bias=False`라 b가 없음)

이 두 텐서는 `requires_grad=True`로 설정돼서 **학습 가능한 파라미터**로 등록됨. 옵티마이저가 학습 중에 이 값들을 업데이트한다.

### 3-3. 왜 `d_model → d_model`로 차원이 같은가?

차원은 같지만 **방향(의미)이 다른 공간**으로 옮기는 것.

같은 768차원 공간이지만:
- W_query를 통과한 벡터는 "질문 공간"에 위치
- W_key를 통과한 벡터는 "검색 키 공간"에 위치
- W_value를 통과한 벡터는 "전달할 내용 공간"에 위치

학습이 진행되면서 각 W가 자기 역할에 맞는 방향으로 회전·변형되도록 조정됨.

### 3-4. 네 개의 레이어 각각의 역할

| 레이어 | 입력 → 출력 | 역할 |
|---|---|---|
| `W_query` | x → Q | "내가 찾고 싶은 것"을 표현 |
| `W_key` | x → K | "내가 가진 정보의 색인"을 표현 |
| `W_value` | x → V | "실제로 전달할 내용"을 표현 |
| `out_proj` | concat된 head → 최종 출력 | 여러 head 결과를 섞어서 다음 레이어로 |

**`out_proj`만 다른 점:** Q/K/V는 attention 계산을 위한 **준비** 단계이고, `out_proj`는 attention이 끝난 뒤 결과를 **정리**해서 내보내는 단계. 그래서 보통 `bias=True` (기본값)로 둠.

### 3-5. `bias=qkv_bias=False`인 이유

GPT 같은 모델에서는 Q/K/V projection에 편향을 빼는 게 관례. 이유:
- 편향은 attention score 계산 시 어차피 softmax에 의해 상쇄되는 경우가 많음
- 파라미터 수를 줄이고 계산을 약간 빠르게 함
- 경험적으로 성능 차이가 거의 없음

반면 `out_proj`는 일반 Linear처럼 bias가 있어도 무방.

### 3-6. `self.`을 붙이는 이유

`self.`을 붙여서 인스턴스 속성으로 만들면, `nn.Module`이 **자동으로 파라미터를 추적**한다. 그래서:
- `model.parameters()` 호출 시 W_query의 가중치가 자동으로 포함됨
- `model.to('cuda')` 호출 시 GPU로 함께 이동
- `model.state_dict()`로 저장 시 함께 저장

`self.` 없이 그냥 `W_query = nn.Linear(...)`로 두면 지역 변수로 사라져서 학습 안 됨.

---

## 4. Dropout과 Causal Mask 준비

```python
self.dropout = nn.Dropout(drop_rate)
self.register_buffer("mask", torch.empty(0, 0, dtype=torch.bool))
```

### 4-1. Dropout이란?

학습 중에 **일부 뉴런의 출력을 무작위로 0으로 만드는** 기법. `drop_rate=0.1`이면 10%의 값을 0으로 바꿈.

```
원래:    [0.3, 0.5, 0.2, 0.8, 0.1, 0.4, ...]
Dropout: [0.3, 0.0, 0.2, 0.8, 0.0, 0.4, ...]
```

### 4-2. 왜 일부러 정보를 버리나?

**과적합(overfitting) 방지.**

비유: 시험 준비할 때 매번 같은 친구들과만 공부하면 그 친구들 설명에만 의존하게 된다. Dropout은 "오늘은 너 빠져" 하면서 모델이 **특정 뉴런에 과하게 의존하지 않도록** 강제한다.

### 4-3. 학습/평가 모드 동작 차이

| 모드 | 동작 |
|---|---|
| `model.train()` | Dropout이 실제로 적용됨 |
| `model.eval()` | Dropout이 자동으로 꺼짐 (모든 값 통과) |

추론할 때는 무작위성이 있으면 안 되니까 PyTorch가 알아서 꺼준다.

### 4-4. Buffer란?

`nn.Module`에는 두 종류의 텐서를 저장할 수 있다.

| 종류 | 학습됨? | 저장됨? | GPU 이동? |
|---|---|---|---|
| **Parameter** | O | O | O |
| **Buffer** | X | O | O |

**버퍼 = 학습은 안 되지만 모델의 일부로 함께 다뤄야 하는 텐서.**

### 4-5. 왜 그냥 `self.mask = ...`로 두지 않나?

평범한 속성으로 두면:

```python
self.mask = torch.ones(100, 100)  # CPU에 만들어짐
model.to('cuda')  # ← mask는 안 따라감! 여전히 CPU
```

→ forward에서 GPU 텐서 × CPU 텐서 충돌로 에러 발생.

`register_buffer`로 등록하면 **`.to(device)` 호출 시 자동으로 함께 이동**. `state_dict()`로 저장할 때도 함께 저장됨.

### 4-6. Causal mask가 뭔가

GPT는 **이전 토큰만 보고 다음 토큰을 예측**하도록 학습. 토큰 3을 처리할 때 토큰 4, 5, 6은 볼 수 없어야 함 (커닝 방지).

mask는 불리언 행렬 (seq_len=4 예시):

```
       토큰0  토큰1  토큰2  토큰3
토큰0 [ F     T     T     T  ]
토큰1 [ F     F     T     T  ]
토큰2 [ F     F     F     T  ]
토큰3 [ F     F     F     F  ]
```

`True`인 위치를 나중에 `-inf`로 채우면, softmax 후 그 자리는 0이 되어 "안 본 셈"이 됨.

### 4-7. 왜 처음에 `torch.empty(0, 0)`?

**비어 있는 0x0 텐서로 시작.** 이유:
- 모델 생성 시점에는 입력 시퀀스 길이를 모름
- 미리 큰 mask를 만들면 메모리 낭비
- 그래서 일단 자리만 잡아두고, forward에서 실제 필요한 크기로 만들어 채움 (지연 초기화)

---

## 5. Python/PyTorch 문법 다지기

### 5-1. `nn.Dropout` 문법 분해

```python
self.dropout = nn.Dropout(drop_rate)
```

- `nn.Dropout(...)`은 클래스 호출 → 인스턴스 생성
- `self.dropout`은 이 객체를 인스턴스 속성으로 저장
- 만들어진 객체는 `self.dropout(x)`처럼 호출 가능 — `nn.Module`이 `__call__`을 정의해뒀기 때문

### 5-2. `register_buffer` 문법 분해

```python
self.register_buffer("mask", torch.empty(0, 0, dtype=torch.bool))
```

- `register_buffer`는 메서드 호출 (할당이 아님)
- 첫 번째 인자(문자열)가 속성명이 됨 → 이후 `self.mask`로 접근 가능
- 메서드를 거쳐야 PyTorch가 "이건 모듈의 일부 버퍼야"라고 인식

### 5-3. 왜 함수가 아니라 클래스인가? (상태)

**함수는 상태가 없음 (stateless):** 입력만 받아서 출력을 내놓고 끝.

**클래스는 상태를 가짐 (stateful):** 객체 안에 `self.뭐든`으로 정보를 저장하고, 호출할 때마다 그 상태에 접근/변경.

Dropout은 학습/추론 모드에 따라 동작이 달라야 함. 이 모드를 어딘가에 **저장**해야 하므로 클래스가 적합. `nn.Module`이 내부적으로 `self.training` 불리언을 가지고 있어서 `model.train()`/`model.eval()` 한 번에 모든 하위 모듈의 상태가 바뀐다.

> 참고: `torch.nn.functional.dropout(x, p, training)` 같은 함수형도 있음. 상태 관리가 필요 없을 때는 함수형도 OK.

### 5-4. 속성(attribute)

객체에 **딸려 있는 데이터(또는 기능)**. `객체.이름`으로 접근.

```python
class Dog:
    def __init__(self, name, age):
        self.name = name    # 속성
        self.age = age      # 속성
```

| 종류 | 예 | 설명 |
|---|---|---|
| 데이터 속성 | `bobby.name` | 값을 저장 |
| 메서드 속성 | `bobby.bark()` | 함수를 저장 (호출 가능) |

### 5-5. 키워드 인자

함수에 값을 전달하는 두 가지 방식:

```python
greet("승철", 25, "서울")            # 위치 인자: 순서로 매칭
greet(name="승철", age=25, city="서울")  # 키워드 인자: 이름으로 매칭
```

**키워드 인자의 장점:**
1. 순서 무관
2. 의미 명확
3. 일부 인자만 골라서 전달 가능

**규칙:** 키워드 인자는 위치 인자보다 뒤에 와야 함.

`attention.py`에서:
```python
nn.Linear(self.d_model, self.d_model, bias=qkv_bias)
torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
torch.triu(텐서, diagonal=1)
```

핵심 인자는 위치로, **설정 옵션 성격의 인자는 키워드로** 넘기는 게 관례.

### 5-6. `self`의 두 가지 역할

`self`는 단순히 **"이 메서드를 호출한 객체 자신"**을 가리키는 변수.

| 표현 | 의미 |
|---|---|
| `self.name = "바비"` | 이 객체에 name 속성을 만들거나 덮어쓰기 |
| `self.name` | 이 객체의 name 속성을 읽기 |
| `self.bark()` | 이 객체의 bark 메서드를 호출 |
| `self.register_buffer(...)` | 이 객체의 register_buffer 메서드를 호출 |

핵심: **`self.` 뒤에 오는 것이 데이터냐 함수냐, 그리고 `=`로 대입하느냐 `()`로 호출하느냐**에 따라 의미가 달라진다.

### 5-7. 객체 = 인스턴스

| 용어 | 의미 |
|---|---|
| **클래스** | 설계도 (틀) |
| **인스턴스** | 설계도로 찍어낸 실제 물건 |
| **객체** | 인스턴스와 거의 같은 말 (메모리에 만들어진 실체) |

```python
attn1 = MultiHeadAttention(d_model=768, n_heads=12)
attn2 = MultiHeadAttention(d_model=512, n_heads=8)
```

같은 클래스로 만든 **두 개의 다른 객체**. 각자 자기만의 `self.d_model`, `self.W_query` 등을 가짐. 서로 영향 안 줌.

### 5-8. `super().__init__()`

**부모 클래스의 `__init__` 메서드를 호출해서, 부모가 해야 할 초기화 작업을 먼저 처리.**

`nn.Module`을 상속받으면 부모가 만들어야 할 내부 자료구조(`_parameters`, `_buffers`, `_modules`, `training` 등)가 있다. 이게 없으면 PyTorch가 파라미터/버퍼/하위모듈을 인식 못 함 → 에러.

**자동 vs 명시적:**
- 자식이 `__init__`을 정의 안 하면 → 부모 것이 자동 실행
- 자식이 `__init__`을 정의하면 → 부모 것은 **명시적으로** 호출해야 함

객체 생성 시(`MultiHeadAttention(768, 12)`)에는 신경 쓸 필요 없음. `super()`는 클래스 정의 안에 적어둔 거라 알아서 실행된다.

### 5-9. `super`에만 `self.`이 안 붙는 이유

`self.뭐든`은 **"이 객체에 속한 것"**을 가리킬 때만. `super`는 객체 속성이 아니라 **파이썬 내장 함수**라서 `self.`이 안 붙는다.

`super()`는 호출된 위치의 컨텍스트를 자동으로 읽어 self를 알아낸다. 파이썬 3 이전엔 `super(MultiHeadAttention, self).__init__()` 이렇게 명시했어야 함.

### 5-10. `self`를 쓸 때 vs 안 쓸 때

| 대상 | self 필요? |
|---|---|
| 객체의 속성 (저장된 데이터) | `self.name` |
| 객체의 메서드 (같은 클래스의 다른 함수) | `self.bark()` |
| 지역 변수 (함수 안에서만 쓰는 임시 변수) | 그냥 `temp` |
| 매개변수 (인자로 받은 값) | 그냥 `name` (저장하려면 `self.name = name`) |
| 내장 함수 / 외부 함수 | `print()`, `len()`, `super()` |
| 메서드 정의 시 첫 매개변수 | `def 메서드(self, ...)` |

**핵심 규칙:** "함수가 끝나도 살아남아야 한다면 `self.`로 객체에 저장. 그 함수 안에서만 쓰고 버릴 거면 그냥 변수."

### 5-11. `__init__` 시그니처

```python
def __init__(
    self,
    d_model: int,
    n_heads: int,
    drop_rate: float = 0.1,
    qkv_bias: bool = False,
):
```

- `self`: 자동으로 채워짐
- `d_model: int`, `n_heads: int`: 필수 인자 (기본값 없음)
- `drop_rate: float = 0.1`, `qkv_bias: bool = False`: 옵션 인자 (기본값 있음)

**순서 규칙:** 필수 인자가 앞에, 기본값 있는 인자가 뒤에.

| 메서드 | 언제 실행 | 인자의 성격 |
|---|---|---|
| `__init__` | 객체 생성 시 1회 | **모델 구조 설정** (차원, head 수 등) |
| `forward` | 호출할 때마다 | **실제 처리할 데이터** |

---

## 6. forward 메서드 시그니처

```python
def forward(
    self,
    x: torch.Tensor,
    causal_mask: bool = True,
    return_attention_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
```

| 부분 | 의미 |
|---|---|
| `이름: 타입` | 타입 힌트 (강제 아님) |
| `이름: 타입 = 값` | 타입 힌트 + 기본값 |
| `-> 타입` | 반환 타입 힌트 |
| `A \| B` | A 또는 B (Python 3.10+) |
| `tuple[A, B]` | A와 B를 담은 튜플 |

**반환:**
- `return_attention_weights=False`: `torch.Tensor` 하나
- `return_attention_weights=True`: `tuple[torch.Tensor, torch.Tensor]`

**호출 관례:** `model.forward(x)`가 아니라 `model(x)`로 호출. `nn.Module.__call__`이 hook 등 부가 작업을 처리한 뒤 `forward`를 호출하기 때문.

---

## 7. 입력 모양 분해와 Q/K/V 계산

```python
batch_size, seq_len, _ = x.shape  # x: (B, T, C)

query = self.W_query(x)
key = self.W_key(x)
value = self.W_value(x)
```

### 7-1. `(B, T, C)` 표기

PyTorch와 트랜스포머 코드의 표준 약자.

| 약자 | 풀네임 | 의미 | 이 코드에서 |
|---|---|---|---|
| **B** | **B**atch size | 한 번에 처리할 문장 개수 | `batch_size` |
| **T** | **T**ime steps (sequence length) | 시퀀스 안의 토큰 개수 | `seq_len` |
| **C** | **C**hannels (dimension) | 각 토큰을 표현하는 벡터의 차원 | `d_model` (768) |

- **T = Time**: RNN 시대 용어. 시퀀스를 "시간에 따라 들어오는 토큰들"로 봤기 때문.
- **C = Channels**: CNN(이미지)에서 온 용어. 트랜스포머에서는 "각 토큰을 몇 차원 벡터로 표현했는가". 다른 표기로 `D`, `E`, `d_model`도 씀.

### 7-2. 튜플 언패킹과 `_`

```python
batch_size, seq_len, _ = x.shape
```

- 파이썬의 **튜플 언패킹** 문법. 오른쪽 튜플의 값을 왼쪽 변수들에 순서대로 꺼내 넣음
- `_`(언더스코어)는 파이썬 관례로 **"이 값은 받긴 하지만 안 쓸 거야"**라는 표시
- 세 번째 값(`d_model`)은 이미 `self.d_model`로 알고 있으니 안 받음

### 7-3. Q/K/V 계산

`self.W_query(x)`는:
1. `self.W_query` → 속성 접근. `nn.Linear` 객체를 꺼냄
2. `(x)` → 그 객체를 함수처럼 호출
3. 내부적으로 `nn.Module.__call__`이 실행되고, 그게 다시 `Linear.forward(x)`를 호출
4. `Linear.forward`는 `x @ W.T + b` 계산을 수행

**모양은 그대로 유지:** `(B, T, 768)` → `(B, T, 768)`

---

## 8. 선형 변환의 동작 원리 (심화)

### 8-1. 왜 마지막 차원에만 행렬곱이 적용되는가

**핵심 원리:** `nn.Linear`는 본질적으로 "벡터 하나를 변환하는 함수".

`nn.Linear(768, 768)`이 정의하는 변환은:
```
벡터 하나(768차원) → 벡터 하나(768차원)
```

`(4, 10, 768)` 모양 텐서는 "768차원 벡터가 40개 있는 것"으로 볼 수 있다. PyTorch는 이걸 **"40개 벡터 각각에 같은 변환을 독립적으로 적용"**한다.

**브로드캐스팅 규칙:**
```
x.shape   = (4, 10, 768)
W.T.shape = (768, 768)

x @ W.T:
   마지막 두 차원 → (10, 768) @ (768, 768) = (10, 768)  ← 진짜 행렬곱
   앞 차원       → 4가 그대로 유지
   
결과: (4, 10, 768)
```

**비유:** 세무사가 "소득 → 세금" 계산 공식을 하나 가지고 있다고 하면, 사람 한 명이든 1000명이든 회사 100개의 직원이든 **같은 공식을 각자에게 따로 적용**한다.

### 8-2. "Query 공간으로 변환한다"는 것의 의미

**시각적 (회전 + 늘이기):** 행렬곱은 기하학적으로 **벡터를 회전시키고, 늘이고 줄이는** 작업. 같은 단어들이 **다른 좌표로 옮겨가는** 것.

**직관 (관점 바꾸기):** 같은 사물도 어떤 관점에서 보느냐에 따라 다른 정보가 보임.

| 관점(공간) | 무엇을 강조 |
|---|---|
| 임베딩 공간 (원본) | "이 단어의 일반적 의미" |
| Query 공간 | "이 단어가 **찾고 싶어 하는** 게 뭔지" |
| Key 공간 | "이 단어가 **검색되기 위한** 표시" |
| Value 공간 | "이 단어가 **실제로 전달할 정보**" |

예: "그것"이라는 대명사가 있다면:
- **Query 공간**에서는 "나는 명사를 찾고 싶어!" 같은 신호가 강조됨
- **Key 공간**에서 다른 명사들은 "나는 명사야"라는 신호가 강조됨

**신경망 맥락 (부분 공간으로의 사영):** 원본 768차원 공간 안에는 단어의 의미, 품사, 위치 정보, 감정 등 **온갖 정보가 섞여** 있다. W_query는 이 중에서 **"질문하는 데 유용한 정보"만 강조**하고 나머지는 약화시키는 일종의 필터.

### 8-3. W가 "역할에 맞는 방향"으로 조정되는 원리

**학습의 큰 그림:**
1. 예측 → 2. 채점(loss) → 3. 반성(미분) → 4. 수정 → 반복

처음에는 W_query, W_key, W_value 모두 **무작위 값**으로 시작. 학습이 진행되면서 손실을 줄이는 방향으로 조정됨.

### 8-4. 계산 구조가 학습을 유도하는 원리 (Chain Rule)

**핵심:** 같은 손실에서 나온 미분 신호라도, **각 W가 계산식의 어디에 위치하느냐에 따라 받는 gradient의 형태가 달라지고**, 그 차이가 곧 역할 차이를 만든다.

**Attention 계산식:**
```
1단계: Q = x @ W_query
2단계: K = x @ W_key
3단계: V = x @ W_value
4단계: S = Q @ K.T / √d
5단계: A = softmax(S)
6단계: O = A @ V
7단계: L = Loss(O, 정답)
```

**각 W의 gradient 경로:**

| 가중치 | 경로 | gradient에 곱해지는 핵심 | 학습되는 것 |
|---|---|---|---|
| W_value | L→O→V→W | **A** (attention weight) | "전달할 내용을 잘 담기" |
| W_query | L→O→A→S→Q→W | **K, V** | "올바른 K를 찾는 질문" |
| W_key | L→O→A→S→K→W | **Q, V** | "올바른 Q에게 발견되는 신호" |

**비대칭성의 핵심:** `S = Q @ K.T`에서 행렬곱은 교환법칙이 성립하지 않으므로 Q와 K의 위치가 다르다. 이 비대칭성 때문에 학습 신호도 비대칭으로 흘러간다.

**역할 분담이 자동으로 일어나는 이유:** 만약 W_query와 W_key가 같은 일을 한다면, 둘 중 하나가 손실에 더 큰 영향을 미치게 되고, gradient도 그쪽이 더 강하게 받는다. 시스템 전체가 효율적인 균형점을 찾으면서 **각자가 자기 위치에 가장 잘 맞는 일**을 맡게 된다.

**중요:** 역할은 강제된 것이 아니라 **계산 구조와 손실 함수가 자연스럽게 유도한 것**이다. 코드 어디에도 "W_query야, 너는 질문 역할을 해라!"라고 명시한 곳은 없다.

**비유 (오케스트라):** 단원 셋이 같은 곡을 처음 연습. 지휘자(손실 함수)가 피드백을 줌. 바이올린(W_query), 첼로(W_key), 팀파니(W_value) 각자 다른 악기와 다른 위치라서 같은 피드백을 받아도 서로 다른 연주를 익히게 된다.

---

## 9. Head 분리: view + transpose

```python
query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
```

### 9-1. 왜 head로 쪼개는가? (Multi-Head의 의미)

```
Single Head:  768차원으로 한 가지 관점의 attention 1번
Multi Head:   64차원짜리 12개로 12가지 관점의 attention 12번
```

각 head가 **서로 다른 관계 패턴**을 학습하길 기대:
- 어떤 head는 "주어-동사" 관계에 집중
- 어떤 head는 "대명사-선행사" 관계에 집중
- 어떤 head는 "위치적 인접성"에 집중

### 9-2. `view()`: 모양 재해석

`view`는 **데이터는 그대로 두고 모양만 다시 해석**하는 함수.

```
원래 모양:    (4, 10, 768)
view 후:     (4, 10, 12, 64)
```

총 원소 개수는 같아야 함: `4 × 10 × 768 = 4 × 10 × 12 × 64 = 307,200`

**메모리는 안 움직임.** view는 메모리에 있는 숫자를 옮기지 않고 "보는 방식"만 바꾼다. 그래서 매우 빠름 (O(1)).

### 9-3. `transpose(1, 2)`: 차원 순서 바꾸기

```
view 후:        (batch_size, seq_len, n_heads, head_dim)
                     0           1        2         3
                                 └────┬────┘
                                    바꿈
transpose 후:   (batch_size, n_heads, seq_len, head_dim)
```

### 9-4. 왜 두 단계로 하나? (view만으로 안 되는 이유)

원본 데이터의 메모리 배치:
```
[토큰0의 768개, 토큰1의 768개, 토큰2의 768개, ...]
```

`view(B, T, H, D)`는 메모리 순서와 일치하니까 가능. 하지만 `(B, H, T, D)`는 메모리상에서 "head 0의 모든 토큰, head 1의 모든 토큰, ..." 순서여야 함. 이건 원본과 순서가 달라서 **실제로 데이터를 옮겨야** 한다.

`transpose`는 view와 달리 **차원의 순서를 진짜로 바꿀 수 있는** 연산.

### 9-5. 벡터 차원이 축소된 게 아님

**정답은 "아니오".** 정보가 줄어든 게 아니라 **나뉜** 것.

```
원래 토큰 벡터:  768차원 한 덩어리
                     ↓
head 분리 후:    64차원 × 12개
```

총 정보량은 그대로: `768 = 12 × 64`. **숫자 개수는 똑같고, 그저 어떻게 묶어서 보느냐**만 달라진 것.

**비유:** 피자 한 판(768)을 12조각(64씩)으로 자른 것. 피자의 총량은 그대로, 다만 한 덩어리로 먹지 않고 12명이 한 조각씩 따로 분석하고 나중에 합친다.

---

## 10. PyTorch 행렬곱 규칙

**황금 규칙:**
```
A @ B 에서 PyTorch는:
- 마지막 두 차원 → "실제 행렬곱"
- 그 앞의 모든 차원 → "배치 차원" (각자 따로 계산)
```

### 10-1. 4D 텐서 행렬곱 예시

```python
A.shape = (10, 7, 3, 4)
B.shape = (10, 7, 4, 5)

A @ B → (10, 7, 3, 5)
```

내부 동작:
```
for i in range(10):
    for j in range(7):
        결과[i][j] = A[i][j] @ B[i][j]
```

앞의 (10, 7)은 모두 배치 차원, 마지막 두 차원만 진짜 행렬곱. **총 70번의 독립적인 행렬곱이 자동으로 병렬 처리됨.**

### 10-2. 왜 (B, n_heads, T, head_dim) 모양이 필요한가

attention에서 우리가 원하는 것: "각 batch의 각 head마다, (T, head_dim) × (head_dim, T) → (T, T) 행렬곱"

**❌ 잘못된 모양: `(B, T, n_heads, head_dim)`**
```
앞의 (B, T)를 배치로 보고
마지막 두 차원 (n_heads, head_dim) × (head_dim, n_heads) → (n_heads, n_heads)
```
→ "각 batch의 각 토큰마다, head들끼리의 행렬곱" — 우리가 원하는 게 아님!

**✓ 올바른 모양: `(B, n_heads, T, head_dim)`**
```
앞의 (B, n_heads)를 배치로 보고
마지막 두 차원 (T, head_dim) × (head_dim, T) → (T, T)
```
→ "각 batch의 각 head마다, 토큰들 사이의 attention score"

### 10-3. "n_heads를 배치 차원의 일부로 만든다"의 의미

원래 배치 차원은 `B` 하나. transpose로 `n_heads`를 앞쪽 두 번째 자리로 옮기면, PyTorch 입장에서 앞쪽 두 차원 `(B, n_heads)` 모두 **배치로 취급**된다.

→ **"B × n_heads 개의 작은 attention 계산을 한꺼번에 처리"**

예: B=4, n_heads=12이면 → 4×12 = 48개의 attention 계산이 한 번의 `@` 연산으로 병렬 처리.

---

## 11. Attention Score 계산

```python
attn_score = query @ key.transpose(-2, -1)
attn_score = attn_score / (self.head_dim ** 0.5)
```

### 11-1. `query @ key.transpose(-2, -1)` 분해

**transpose(-2, -1):** 마지막 두 차원을 서로 바꿈
```
key:           (B, H, T, head_dim)
key.transpose: (B, H, head_dim, T)
```

**행렬곱:**
```
query:           (B, H, T, head_dim)
key.transpose:   (B, H, head_dim, T)
                          ↑       ↑
                          만남     결과의 마지막 차원
결과: (B, H, T, T)
```

**결과의 의미:**
```
attn_score[b][h][i][j] = "batch b, head h에서, i번째 토큰의 query와 j번째 토큰의 key가 얼마나 닮았는지"
```

### 11-2. 왜 곱하면 "관련도"가 되나? (내적의 의미)

행렬곱은 결국 **내적(dot product)**의 모음.

```
attn_score[i][j] = query[i] · key[j]
                 = q[i][0]×k[j][0] + ... + q[i][63]×k[j][63]
```

두 벡터의 내적은 **방향이 비슷할수록 큰 값**:
- 같은 방향: 내적 → 큰 양수 (관련 높음)
- 직각: 내적 → 0 (관련 없음)
- 반대 방향: 내적 → 큰 음수 (반대 관계)

### 11-3. 스케일링 (`/ √head_dim`)

```python
attn_score = attn_score / (self.head_dim ** 0.5)
```

`** 0.5`는 제곱근. head_dim=64라면 `√64 = 8`로 나눈다.

**왜 나누나? (softmax의 함정):** softmax는 큰 값을 더 크게, 작은 값을 더 작게 만든다. 입력 값들의 **크기가 너무 크면** softmax 출력이 거의 0과 1로 극단화된다.

```
softmax([1, 2, 3])         → [0.09, 0.24, 0.67]    ← 부드러움
softmax([10, 20, 30])      → [0.00, 0.00, 1.00]    ← 거의 1에 다 몰림
```

극단화되면 (1) 표현력 손실, (2) gradient 소실 문제 발생.

**왜 하필 √head_dim?:** 내적의 값은 통계적으로 차원 수가 커질수록 분산이 커진다. Q와 K의 각 원소가 평균 0, 분산 1인 독립 분포라고 가정하면, 둘의 내적은 평균 0, **분산 head_dim**을 갖는다. 표준편차는 `√head_dim`. → 표준편차로 나누면 분산이 1로 정규화.

→ 이 스케일링 때문에 attention의 정식 이름이 **"Scaled Dot-Product Attention"** (Vaswani 2017).

---

## 12. Causal Mask 적용

```python
if causal_mask:
    if self.mask.size(0) < seq_len or self.mask.device != x.device:
        self.mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
    attn_score = attn_score.masked_fill(self.mask[:seq_len, :seq_len], float("-inf"))
```

### 12-1. 왜 미래를 가려야 하나

GPT의 학습 목표: **이전 토큰들만 보고 다음 토큰을 예측.** 문장 전체를 한 번에 학습할 때 각 위치가 미래 토큰을 못 보도록 막아야 함 (커닝 방지).

"causal(인과적)" 또는 "autoregressive(자기회귀)"라고 부름.

### 12-2. mask 행렬 (seq_len=4)

```
       k0    k1    k2    k3
q0  [  F     T     T     T  ]   ← 토큰 0은 1,2,3을 가려야 함
q1  [  F     F     T     T  ]
q2  [  F     F     F     T  ]
q3  [  F     F     F     F  ]   ← 토큰 3은 다 볼 수 있음
```

- **True**: 가려야 할 위치 (미래)
- **False**: 볼 수 있는 위치 (자신 + 과거)

상삼각(upper triangular) 모양이라 `torch.triu`로 만든다.

### 12-3. 캐싱 조건

```python
if self.mask.size(0) < seq_len or self.mask.device != x.device:
```

두 가지 경우 새로 만듦:
1. **현재 mask 크기가 부족할 때**
2. **device가 다를 때** (모델 GPU 이동 후)

조건 미충족 시 **재사용**. 매번 새로 만들면 비효율.

### 12-4. `torch.triu(..., diagonal=1)`

`triu`는 **상삼각(upper triangular)** 만드는 함수. `diagonal=1`은 "대각선보다 한 칸 위부터 남긴다".

```
diagonal=0: 대각선 포함 위쪽       diagonal=1: 대각선 제외 위쪽
[ T T T T ]                         [ 0 T T T ]
[ 0 T T T ]                         [ 0 0 T T ]
[ 0 0 T T ]                         [ 0 0 0 T ]
[ 0 0 0 T ]                         [ 0 0 0 0 ]
```

`diagonal=1`인 이유: **자기 자신은 볼 수 있어야** 하기 때문. 대각선 위치(q0-k0 등)는 가리면 안 됨.

### 12-5. `masked_fill`과 `-inf`

```python
attn_score.masked_fill(self.mask[:seq_len, :seq_len], float("-inf"))
```

- mask가 True인 자리 → `-inf`로 덮어쓰기
- mask가 False인 자리 → 원래 값 유지

**왜 `-inf`?** 다음 단계 softmax: `exp(-inf) = 0`. → -inf 자리는 softmax 출력에서 **정확히 0**이 되어 완벽하게 "안 본 셈"이 됨. 0으로 채우면 `exp(0) = 1`이 되어 여전히 attention이 흘러간다.

### 12-6. `self.mask[:seq_len, :seq_len]` 슬라이싱

저장된 mask가 더 클 수도 있어서 (예: 이전에 길이 100짜리, 이번엔 10) **현재 필요한 부분만 잘라 씀.**

---

## 13. Softmax와 Dropout

```python
attn_weights = torch.softmax(attn_score, dim=-1)
attn_weights = self.dropout(attn_weights)
```

### 13-1. softmax

**임의의 실수들을 "합이 1인 양수 값들"로 바꿈** (확률 분포로 변환).

수식: `softmax(x_i) = exp(x_i) / Σ exp(x_j)`

**핵심 성질:**
1. 모두 양수가 됨
2. 합이 정확히 1
3. 큰 값은 더 크게, 작은 값은 더 작게

```
softmax([-inf, 2])  → [0.00, 1.00]    ← -inf는 정확히 0
```

→ causal mask로 채운 `-inf` 자리가 여기서 정확히 0이 되어 미래 토큰이 무시된다.

### 13-2. `dim=-1`의 의미

```
attn_score.shape = (B, H, T, T)
                              ↑
                          마지막 차원 (key 축)
```

마지막 차원은 "한 query 토큰이 본 **모든 key 토큰의 점수들**". `dim=-1`이면 행마다 softmax 적용.

```
query 1이 본 점수:  [ 0.8   2.1   -∞   -∞ ]
                        ↓ softmax (dim=-1)
              →     [ 0.21  0.79  0.0  0.0 ]
                        합 = 1.0 (확률 분포)
```

**각 query 토큰이 다른 토큰들에 얼마나 주의를 기울일지의 확률 분포** 완성.

### 13-3. attention weights에 dropout

**무작위로 일부 값을 0으로 만들고, 남은 값은 보정 계수로 키움 (inverted dropout).**

```
원래:    [0.30, 0.20, 0.15, 0.10, 0.05, 0.20]
                            ↓ dropout (10%)
적용 후: [0.33, 0.00, 0.17, 0.11, 0.06, 0.22]
              ↑   드롭됨   ↑
              나머지는 1/(1-0.1) ≈ 1.11배 보정
```

**보정 이유 (학습/추론 일관성):** 학습 시 일부를 0으로 만들면 전체 합이 줄어드니, 남은 값들을 키워서 **평균 출력 크기를 일정하게 유지.**

**왜 attention weights에 적용?** 특정 토큰 관계에 과도하게 의존하는 것을 방지. "토큰 A는 항상 토큰 B만 보면 돼"라는 식의 빈약한 패턴에 갇히지 않도록.

---

## 14. 출력 생성

```python
context_vector = attn_weights @ value
context_vector = context_vector.transpose(1, 2)
context_vector.view(batch_size, seq_len, self.d_model)
context_vector = self.out_proj(context_vector)

return context_vector
```

### 14-1. `attn_weights @ value` — 가중합

```
attn_weights:  (B, H, T, T)         ← "어디를 얼마나 볼지" 확률 분포
value:         (B, H, T, head_dim)  ← "전달할 내용"
                              
matmul (T, T) @ (T, head_dim) → (T, head_dim)

context_vector: (B, H, T, head_dim)
```

**의미:**
```
context[i] = attn_weights[i][0] × value[0]
           + attn_weights[i][1] × value[1]
           + ...
```

**"내가 주목한 비율대로 다른 토큰들의 value를 섞은 것"**. 원래는 자기 토큰의 임베딩이었는데, 이제는 **문맥(다른 토큰들의 정보)이 녹아든 표현**으로 진화.

**비유:** "학생들이 노트를 공유하는 회의". attn_weights = 친구들 노트에 대한 신뢰도, value = 각 친구의 노트 내용, context = 신뢰도에 비례해 친구들 노트를 합친 내 종합 노트.

### 14-2. `transpose(1, 2)` — head를 토큰 옆으로

```
처리 전: (B, H, T, head_dim)   ← head가 앞쪽 (병렬 계산용 배치)
처리 후: (B, T, H, head_dim)   ← head를 토큰 옆으로 되돌림
```

처음에 `(B, T, H, D) → (B, H, T, D)`로 옮겼던 것을 거꾸로 되돌리는 단계.

### 14-3. `view(...)` — head들을 합치기

12개 head(각각 64차원)를 다시 768차원 한 덩어리로 합침.
```
처리 전: (B, T, 12, 64)
처리 후: (B, T, 768)
```

### 14-4. `self.out_proj(...)` — 최종 projection

**왜 또 변환하나?** 12개 head 결과를 단순히 이어 붙인 것은 아직 "조각들의 모음". out_proj가 이 조각들을 섞어서 통합된 표현으로 만든다.

```
concat한 직후:  [head0의 결과 | head1의 결과 | ... | head11의 결과]
                  서로 독립적으로 만들어진 조각들
out_proj 후:    [768차원 통합 표현]
                각 차원이 모든 head 정보의 가중합
```

학습이 진행되면서 out_proj가 **"어떤 head 결과를 얼마나 중시할지"**를 배움.

### 14-5. ⚠️ 코드의 버그 두 개

**버그 1: `view` 결과를 안 받음**
```python
context_vector.view(batch_size, seq_len, self.d_model)   # ← 결과 미할당!
```
`view()` 결과가 어디에도 저장되지 않음. 다음 줄에 들어가는 `context_vector`는 여전히 4D.

**버그 2: `view`는 transpose 직후 못 씀**
`transpose`는 텐서를 비연속(non-contiguous) 상태로 만든다. `view`는 **연속 메모리만 처리할 수 있음.**

**수정:**
```python
context_vector = context_vector.transpose(1, 2).contiguous().view(B, T, d_model)
# 또는
context_vector = context_vector.transpose(1, 2).reshape(B, T, d_model)
```

**현재 코드는 실행하면 shape mismatch 에러가 난다.** `out_proj`는 `Linear(768, 768)`인데 들어오는 마지막 차원은 head_dim=64.

### 14-6. 입출력 모양이 같은 이유

입력과 출력 모양이 같으니까(`(B, T, 768)`), 이 블록을 12층, 24층, 96층 쌓아서 GPT 같은 큰 모델을 만들 수 있다. **한 블록의 출력이 다음 블록의 입력으로 그대로 들어감.**

---

## 15. 메모리 심화: view vs reshape, 연속/불연속

### 15-1. view vs reshape 한 줄 차이

**view는 메모리가 연속일 때만 동작하는 빠른 함수, reshape은 필요하면 메모리를 재배치까지 해주는 안전한 함수.**

| 항목 | `view` | `reshape` |
|---|---|---|
| 메모리 연속 시 | 동작 (빠름) | 동작 (빠름) |
| 메모리 비연속 시 | **에러** | **자동 재배치 후 동작** |
| 메모리 재배치 가능성 | 절대 없음 | 필요시 있음 |

### 15-2. 연속 메모리란

**텐서의 원소들이 메모리에서도 논리적 순서대로 일렬로 나열된 상태.**

```
논리적 모양 (3, 4):
[[a, b, c, d],
 [e, f, g, h],
 [i, j, k, l]]

연속 메모리:  [a, b, c, d, e, f, g, h, i, j, k, l]
```

### 15-3. 불연속이 되는 순간

`transpose`, `permute` 같은 연산은 메모리는 안 옮기고 **"보는 순서"만 바꿈.**

```python
x = torch.arange(12).view(3, 4)
x.is_contiguous()           # True

y = x.transpose(0, 1)
y.is_contiguous()           # False
```

`y`는 논리적으로 `(4, 3)` 모양이지만, 메모리는 여전히 원래대로 누워있고 strides만 바꿔서 보고 있는 상태.

**불연속의 정확한 정의:** "논리 순서대로 텐서를 쭉 훑었을 때, 메모리 주소가 일정한 간격(+1칸씩)으로 증가하지 않고 띄엄띄엄 점프하는 상태."

### 15-4. 왜 view만 불연속을 못 다루나

**연산자는 두 부류:**

**부류 A: "똑똑하게 읽는" 연산자 — 불연속 OK**
`@`, `+`, `*`, `softmax`, `sum`, `mean` 등. **strides 정보를 보고 메모리를 올바른 순서로 점프해가며 읽는다.**

**부류 B: "메모리 그대로 재해석"만 하는 연산자 — 연속 필수**
`view`가 거의 유일. 메모리 데이터 자체에 손대지 않고, **"이 메모리 블록을 다른 모양의 텐서로 해석해라"**라고 말할 뿐.

`matmul`이 잘 됐다고 해서 view도 될 거라고 생각하면 안 된다. **두 연산은 작동 원리가 완전히 다름.**

- matmul: 입력 읽고 → 계산 → **새 텐서 생성** (연속 메모리)
- view: 입력의 메모리를 **그대로 다른 모양으로 재해석** (새 메모리 없음)

### 15-5. view는 불연속을 만들지도 않음

view는 연속 텐서만 받아서 연속 텐서를 만든다. 차원의 **재배치**(transpose, permute)가 일어나야 불연속이 되는데, view는 단순 **재해석**만 함.

| 연산 | 메모리 복사 | 순서 변경 | 결과 연속성 |
|---|---|---|---|
| view | X | X | 연속 유지 |
| transpose | X | O | 불연속 됨 |
| reshape | 필요시 O | X | 항상 연속 |
| contiguous | O | X (재정렬) | 연속화 |

### 15-6. "메모리 그대로 보여준다"의 정확한 의미

| 의미 | 맞는가? |
|---|---|
| 메모리 데이터를 **복사 안 한다** | ✓ (view의 핵심 특징) |
| 메모리를 읽는 **순서를 바꾼다** | ✗ (이건 transpose의 일) |

**비유 (도서관):**
- view = "이 책들을 3권씩 4묶음으로 봐" (책 위치 그대로, 묶음 단위만 새로 설정)
- transpose = "이 책들을 세로로 다시 배열해서 봐" (책은 안 옮겼지만 읽는 순서가 바뀜)

---

## 16. 전체 forward 요약

| 단계 | 모양 | 의미 |
|---|---|---|
| 입력 x | `(B, T, 768)` | 토큰 임베딩 |
| Q/K/V projection | `(B, T, 768)` × 3 | 세 가지 역할로 변환 |
| view + transpose | `(B, 12, T, 64)` × 3 | head 분리 |
| Q @ K.T / √d | `(B, 12, T, T)` | attention score |
| + causal mask | `(B, 12, T, T)` | 미래 가리기 |
| softmax + dropout | `(B, 12, T, T)` | 확률 분포로 변환 |
| @ V | `(B, 12, T, 64)` | 문맥 벡터 (head별) |
| transpose + view | `(B, T, 768)` | head 합치기 |
| out_proj | `(B, T, 768)` | 최종 통합 |
| 반환 | `(B, T, 768)` | 입력과 같은 모양 |

---

## Attention의 본질

> **"각 토큰이 다른 토큰들을 자기 필요에 맞게 가중평균해서 새로운 문맥 표현을 만든다."**

이걸 효율적으로 하기 위해:
- **Q/K/V로 역할 분리** → 질문/검색/내용 분담
- **Multi-head** → 여러 관점에서 동시에
- **Causal mask** → 미래 안 보기 (GPT 특성)
- **Scaling, softmax, dropout** → 수치 안정성과 일반화

전체 `attention.py`는 결국 위 한 문장을 **숫자와 행렬로 정확하게 구현한 것**이다.

---

## 사용법 정리

```python
# 1) 모델 생성
attn = MultiHeadAttention(d_model=768, n_heads=12)

# 2) GPU 이동 (선택)
attn = attn.to('cuda')

# 3) 학습 모드 (dropout 켜기)
attn.train()

# 4) 입력 준비
x = torch.randn(4, 10, 768).to('cuda')

# 5) forward 호출
output = attn(x)                    # shape: (4, 10, 768)

# 6) 평가 모드 (dropout 끄기)
attn.eval()
with torch.no_grad():
    output = attn(x)
```

**중요:** `attn.forward(x)`가 아니라 `attn(x)`로 호출. `nn.Module.__call__`이 hook 등을 처리한 뒤 `forward`를 실행한다.
