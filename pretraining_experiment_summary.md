# GPT Pretraining Experiment Report

이 문서는 NSMC LM corpus로 GPT 사전 학습 성능을 개선한 과정을 `가설 -> 액션 -> 결과 -> 판단` 흐름으로 정리한 발표용 기록이다.

정리 기준:

- 대화 중 해석한 실험 결과
- 로컬에 저장된 `summary*.json`, `resume_summary*.json`
- Google Drive 실험 폴더: <https://drive.google.com/drive/u/2/folders/1g05p9z43HNkVHusmh2JirdE4w8aiLfBz>

> 그래프는 Google Drive `thumbnail` 링크로 넣었다. Markdown 미리보기에서 Drive 인증 때문에 이미지가 보이지 않으면, 각 그래프 아래의 `원본 그래프 열기` 링크를 사용한다.

## 1. 실험 목표

목표는 작은 GPT 모델의 사전 학습 validation loss를 낮추고, 이후 감성 분류 fine-tuning에 사용할 checkpoint를 고르는 것이다.

주요 평가 기준은 다음 순서로 보았다.

1. `final_full_val_loss`
2. `best_val_loss_estimate`
3. train-val gap
4. loss curve 안정성
5. 추가 학습 시간 대비 개선폭

## 2. 공통 설정

| 항목 | 값 |
|---|---|
| tokenizer | BPE vocab size 3000 |
| train corpus | 1,379,486 chars |
| validation corpus | 120,560 chars |
| train tokens | 869,883 |
| validation tokens | 76,010 |
| BOS/EOS | 리뷰 line 단위 추가 |
| context length | 64 |
| batch size | 32 |
| model | emb_dim 192, n_heads 8, n_layers 2 |
| dropout | 0.2 |
| optimizer | AdamW |
| weight decay | 0.01 |
| validation eval | full validation loader |

주의:

- 사용자가 구두로 `lr=0.0003`이라고 표현한 구간은 Drive summary 기준으로 `lr=0.003` run이다.
- Drive 폴더명 기준 팀이 공유한 최종 후보는 `BESTPERFORMANCE` 폴더의 `ep20 -> ep30, lr=0.003 resume` checkpoint다.
- 숫자상 최저 `final_full_val_loss`는 이후 `lr=0.0005 resume ep40`이지만, 개선폭과 train-val gap을 고려하면 팀 공유 모델로는 `BESTPERFORMANCE` checkpoint를 채택해도 충분히 방어 가능하다.

## 3. 주요 결과 요약

| 단계 | 설정 | final full val loss | ppl | train loss | val estimate | gap | 판단 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | lr 5e-4, wd 0.01, epoch 15 | 4.9344 | 138.99 | 4.4632 | 4.9419 | 0.4787 | 안정적이지만 성능 한계 |
| lr 상승 | lr 0.01, epoch 15 | 4.8603 | 129.06 | 4.6507 | 4.8617 | 0.2110 | 큰 lr이 예상보다 잘 작동 |
| best base | lr 0.01, epoch 20 | 4.7053 | 110.53 | 4.4088 | 4.7053 | 0.2965 | 이후 resume의 출발점 |
| fresh epoch 30 | lr 0.01, epoch 30 | 4.7939 | 120.78 | 4.4329 | 4.7951 | 0.3623 | 오래 돌리면 튀고 악화 |
| selected resume | ep20 checkpoint -> lr 0.003, ep30 | 4.5626 | 95.83 | 4.1170 | 4.5750 | 0.4580 | 팀 공유 최종 checkpoint |
| resume 2 | ep30 checkpoint -> lr 0.001, ep35 | 4.5516 | 94.78 | 4.0385 | 4.5542 | 0.5157 | 소폭 개선, gap 증가 |
| resume 3 | ep35 checkpoint -> lr 0.0005, ep40 | 4.5419 | 93.87 | 4.0073 | 4.5419 | 0.5346 | 숫자상 최저, 개선폭 작음 |
| warmup/cosine/clip | peak lr 0.01, min lr 5e-4, ep40 | 4.6911 | 108.98 | 3.6916 | 4.6911 | 0.9996 | train만 과하게 내려가 실패 |
| gradient clipping only | lr 0.01, clip 1.0, ep20 | 4.8113 | 123.0 근처 | 4.4594 | 4.8113 | 0.3519 | clipping 단독도 개선 없음 |

## 4. 선형 실험 흐름

### 4.1 초기 가설: 큰 모델과 긴 context가 성능을 올릴 것이다

액션:

- `context_length=128/256`
- `emb_dim=128/192/256`
- `n_layers=2/4/6`
- `batch_size=8/16`

결과:

- `ctx256, emb256, layers6` 계열은 계산량이 크고 validation loss가 빠르게 좋아지지 않았다.
- 작은 데이터셋에서는 모델을 크게 하는 것보다 학습률, dropout, BOS/EOS, resume 전략이 더 중요했다.

판단:

- 이후 실험 범위를 `context_length=64`, `emb_dim=192`, `n_layers=2`, `batch_size=32` 중심으로 좁혔다.

예시 그래프:

![ctx256 large model run](https://drive.google.com/thumbnail?id=1tBQkRv0-MKJizzy41qgImDoOgfGDy4Lc&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1tBQkRv0-MKJizzy41qgImDoOgfGDy4Lc/view)

![ctx128 smaller model run](https://drive.google.com/thumbnail?id=1HiyXeW6VDbaG5nxVAs1cDvLEbqRzIGVm&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1HiyXeW6VDbaG5nxVAs1cDvLEbqRzIGVm/view)

### 4.2 가설: 리뷰별 BOS/EOS를 넣으면 loss가 낮아질 것이다

액션:

- 전체 corpus를 한 덩어리로 encode하지 않고, 리뷰 한 줄마다 `BOS ... EOS`를 추가했다.
- `encode_each_line_with_bos_eos=True`를 사용했다.

결과:

- BOS/EOS를 넣은 뒤 validation loss가 전반적으로 크게 낮아졌다.
- 사용자 관찰 기준으로 약 0.5 수준의 loss 하락이 있었다.

해석:

- NSMC corpus는 짧은 리뷰들이 줄 단위로 이어진 구조다.
- BOS/EOS는 모델에게 리뷰 경계를 알려준다.
- 문장 끝과 새 리뷰 시작을 예측하는 문제가 쉬워져 next-token loss가 낮아진다.

그래프:

![without BOS/EOS](https://drive.google.com/thumbnail?id=1K5A2E3upOGTF6eOGgIFYISAqMDHnd59q&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1K5A2E3upOGTF6eOGgIFYISAqMDHnd59q/view)

![with BOS/EOS](https://drive.google.com/thumbnail?id=1euYRjGug89VKchAo7CYDqoCiT_U2yN6I&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1euYRjGug89VKchAo7CYDqoCiT_U2yN6I/view)

### 4.3 가설: dropout 0.2가 과적합을 줄일 것이다

액션:

- dropout 0.1과 0.2를 비교했다.
- 이후 핵심 실험은 dropout 0.2로 고정했다.

결과:

- dropout 0.2에서 train-val gap이 상대적으로 줄어드는 경향이 있었다.

판단:

- 데이터셋이 작기 때문에 dropout 0.2가 더 안정적이었다.
- 이후 실험에서 dropout은 더 이상 크게 흔들지 않고, learning rate와 resume 전략에 집중했다.

### 4.4 가설: weight decay는 일반화에 도움이 될 것이다

액션:

- AdamW weight decay를 `0.0`, `0.01`, `0.02`로 비교했다.

결과:

| weight decay | final full val loss | 판단 |
|---:|---:|---|
| 0.0 | 4.9390 근처 | baseline보다 약함 |
| 0.01 | 4.9344 | 가장 무난 |
| 0.02 | 4.9374 | 0.01보다 약간 나쁨 |

해석:

- weight decay는 약간 도움이 됐지만, 너무 높이면 개선되지 않았다.
- 이후 `weight_decay=0.01`로 고정했다.

그래프:

![weight decay 0.0](https://drive.google.com/thumbnail?id=1HyCgRUmjmW12CwCqqg82j4TGQatYGTzr&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1HyCgRUmjmW12CwCqqg82j4TGQatYGTzr/view)

![weight decay 0.01](https://drive.google.com/thumbnail?id=1SEHA9zj51JnKLLVsHzeb1VC4FwzsvBMX&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1SEHA9zj51JnKLLVsHzeb1VC4FwzsvBMX/view)

![weight decay 0.02](https://drive.google.com/thumbnail?id=1gNtMh75F3xoentufR8rfro0zLiJfUVpW&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1gNtMh75F3xoentufR8rfro0zLiJfUVpW/view)

### 4.5 가설: lr 5e-4보다 큰 lr이 더 빠르게 좋은 영역에 도달할 수 있다

액션:

- 기존 `lr=5e-4`에서 `lr=0.01`로 크게 올렸다.
- 같은 모델 구조에서 epoch 15, 20을 비교했다.

결과:

| 설정 | final full val loss | train-val gap | 판단 |
|---|---:|---:|---|
| lr 5e-4, epoch 15 | 4.9344 | 0.4787 | 느리고 성능 한계 |
| lr 0.01, epoch 15 | 4.8603 | 0.2110 | 큰 개선 |
| lr 0.01, epoch 20 | 4.7053 | 0.2965 | best base |

해석:

- 작은 GPT와 작은 corpus 조건에서는 `lr=0.01`이 초중반 학습에 효과적이었다.
- train loss를 무작정 낮추는 것보다 validation loss가 빠르게 내려가는 구간을 찾는 것이 중요했다.

그래프:

![lr 0.01 epoch 15](https://drive.google.com/thumbnail?id=1T-w2tDciVZeiNE7wMj3gPyZiNT9mSMAu&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1T-w2tDciVZeiNE7wMj3gPyZiNT9mSMAu/view)

![lr 0.01 epoch 20 BESTBASE](https://drive.google.com/thumbnail?id=1oK2BhpUNOVSGrj__BnV-WyEObCCboG9X&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1oK2BhpUNOVSGrj__BnV-WyEObCCboG9X/view)

### 4.6 반증 실험: lr 0.01로 그냥 30 epoch까지 돌리면 더 좋아질까?

액션:

- 처음부터 `lr=0.01`로 epoch 30까지 학습했다.

결과:

- epoch 20 run: `final_full_val_loss=4.7053`
- fresh epoch 30 run: `final_full_val_loss=4.7939`

판단:

- 단순히 더 오래 돌리면 좋아지는 것이 아니었다.
- 후반부에는 `lr=0.01`이 너무 커서 validation loss가 튀었다.

그래프:

![fresh lr 0.01 epoch 30](https://drive.google.com/thumbnail?id=1IvfKMNqO4XqAZniIH5sNaL_yPolmbRZI&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1IvfKMNqO4XqAZniIH5sNaL_yPolmbRZI/view)

### 4.7 핵심 가설: 좋은 checkpoint에서 lr을 낮춰 이어 학습하면 성능이 더 좋아질 것이다

액션:

- epoch 20 checkpoint를 불러왔다.
- optimizer state는 새로 시작하고, lr을 낮춰 추가 학습했다.

실험:

```text
stage 1: lr 0.01, epoch 1~20
stage 2: lr 0.003, epoch 21~30
```

결과:

- `4.7053 -> 4.5626`
- fresh epoch 30의 `4.7939`보다 훨씬 좋았다.

해석:

- 큰 lr은 초중반 탐색에 좋다.
- 후반부에는 작은 lr로 미세 조정하는 것이 더 좋다.
- 이 전략은 수동 learning rate decay, stage-wise training, continued pretraining에 가깝다.

그래프:

![resume epoch 20 to 30 BESTPERFORMANCE](https://drive.google.com/thumbnail?id=1Br0rVQUqcxSpp3UKSZYQhM6b0pqhQfZX&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1Br0rVQUqcxSpp3UKSZYQhM6b0pqhQfZX/view)

### 4.8 추가 가설: lr을 더 낮추면 더 좋아질 수 있다

액션:

```text
stage 3: ep30 checkpoint -> lr 0.001, epoch 31~35
stage 4: ep35 checkpoint -> lr 0.0005, epoch 36~40
```

결과:

| 단계 | final full val loss | 개선폭 | gap |
|---|---:|---:|---:|
| ep20 -> ep30, lr 0.003 | 4.5626 | 0.1426 개선 | 0.4580 |
| ep30 -> ep35, lr 0.001 | 4.5516 | 0.0110 개선 | 0.5157 |
| ep35 -> ep40, lr 0.0005 | 4.5419 | 0.0097 개선 | 0.5346 |

판단:

- 숫자상 최저는 ep40 resume다.
- 그러나 개선폭이 작아지고 train-val gap은 계속 커졌다.
- 팀 공유 최종 모델로는 `BESTPERFORMANCE` ep30 checkpoint를 쓰는 판단도 합리적이다.

그래프:

![resume epoch 30 to 35](https://drive.google.com/thumbnail?id=1mlRyd94MuZYCXkTZc8jcz52cQfg8DmST&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1mlRyd94MuZYCXkTZc8jcz52cQfg8DmST/view)

![resume epoch 35 to 40](https://drive.google.com/thumbnail?id=1HmyX1RUVB3_4G9uPC8X4fTst11wO0YIN&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1HmyX1RUVB3_4G9uPC8X4fTst11wO0YIN/view)

### 4.9 가설: warmup + cosine decay + gradient clipping이 수동 decay보다 낫지 않을까?

액션:

- `peak_lr=0.01`
- `min_lr=5e-4`
- `warmup_ratio=0.03`
- `grad_clip=1.0`
- epoch 40

결과:

- train loss는 크게 내려갔다.
- validation loss는 현재 best보다 나빴다.
- train-val gap이 크게 커졌다.

수치:

| 설정 | final full val loss | train loss | gap |
|---|---:|---:|---:|
| manual resume ep40 | 4.5419 | 4.0073 | 0.5346 |
| warmup/cosine/clip ep40 | 4.6911 | 3.6916 | 0.9996 |

판단:

- 이 설정의 cosine decay는 중후반 lr을 너무 오래 높게 유지한 것으로 보인다.
- train만 과하게 내려가고 validation은 못 따라왔다.
- 최종 전략으로 채택하지 않았다.

### 4.10 가설: gradient clipping만 남기면 안정화에 도움이 될까?

액션:

- scheduler는 끄고, `grad_clip=1.0`만 적용했다.

결과:

- validation loss가 오히려 더 나빠졌다.
- gradient clipping이 현재 병목은 아니었다.

판단:

- 최종 학습에서는 warmup, cosine decay, gradient clipping 모두 사용하지 않는다.
- 최종으로는 checkpoint resume 기반 수동 lr decay를 채택한다.

그래프:

![gradient clipping only epoch 20](https://drive.google.com/thumbnail?id=1DG5CPiy6zMRaQTlr2vc7hH47_HEYz2up&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1DG5CPiy6zMRaQTlr2vc7hH47_HEYz2up/view)

## 5. 최종 모델 선택

팀 fine-tuning에 넘긴 최종 checkpoint:

```text
Drive folder:
20260603_065413_resume_from_ep20_to_ep30_ctx64_bos_emb192_heads8_layers2_drop0.2_bs32_lr0.003_wd0.01_BESTPERFORMANCE
```

최종 채택 이유:

- from-scratch epoch 20에서 `4.7053`까지 도달
- checkpoint resume으로 `4.5626`까지 큰 폭 개선
- 이후 lr 0.001, 0.0005 resume은 숫자상 조금 더 낮지만 개선폭이 작고 gap이 커짐
- warmup/cosine/clip과 gradient clipping only는 성능 개선 실패

최종 발표용 문장:

```text
초기에는 큰 모델과 긴 context를 시도했지만, validation loss 개선은 제한적이었다.
BOS/EOS를 리뷰 단위로 넣고, Optuna 및 수동 실험을 통해 context_length 64, emb_dim 192, layers 2, dropout 0.2 조합으로 좁혔다.
lr 0.01은 초중반 학습에 효과적이었지만, 같은 lr로 오래 학습하면 validation loss가 튀었다.
따라서 epoch 20 checkpoint에서 learning rate를 0.003으로 낮춰 이어 학습했고, 이 방법이 가장 큰 성능 개선을 만들었다.
warmup/cosine decay와 gradient clipping은 추가 실험했지만 train-val gap이 커지거나 validation loss가 악화되어 최종 모델에는 적용하지 않았다.
```

## 6. Drive 그래프 Appendix

아래 그래프들은 Drive 폴더에서 확인한 loss curve 파일이다. 본문 핵심 흐름 외의 보조 실험도 시간 순서대로 남긴다.

### 6.1 초기 크기/구조 탐색

![20260602 ctx128 bos emb192 layers4 bs16 lr0.0005](https://drive.google.com/thumbnail?id=1WTU4IsxqEdIM1qBhxnlgNn6N0chcae6A&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1WTU4IsxqEdIM1qBhxnlgNn6N0chcae6A/view)

![20260602 ctx128 bos emb192 layers4 bs16 lr0.0003](https://drive.google.com/thumbnail?id=124xda6vkx5zeqyBW8i3ZiZsWsC9GwU-8&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/124xda6vkx5zeqyBW8i3ZiZsWsC9GwU-8/view)

![ctx128 bos emb128 heads4 layers2 lr0.0005 ep10](https://drive.google.com/thumbnail?id=1HiyXeW6VDbaG5nxVAs1cDvLEbqRzIGVm&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1HiyXeW6VDbaG5nxVAs1cDvLEbqRzIGVm/view)

![ctx256 bos emb256 heads8 layers6 lr0.0005 ep10 run A](https://drive.google.com/thumbnail?id=1tBQkRv0-MKJizzy41qgImDoOgfGDy4Lc&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1tBQkRv0-MKJizzy41qgImDoOgfGDy4Lc/view)

![ctx256 bos emb256 heads8 layers6 lr0.0005 ep10 run B](https://drive.google.com/thumbnail?id=1_q3MvX5S6tXUfGlX0zcBcFUnI9mK42rX&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1_q3MvX5S6tXUfGlX0zcBcFUnI9mK42rX/view)

![ctx128 bos emb128 heads4 layers2 bs8 lr0.0005 ep20](https://drive.google.com/thumbnail?id=1_2x4uqfV_zwamHHb56rb-fG1kSK-lY8l&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1_2x4uqfV_zwamHHb56rb-fG1kSK-lY8l/view)

### 6.2 BOS/EOS 비교

![ctx128 no BOS/EOS emb192 layers4 bs16 lr0.0005](https://drive.google.com/thumbnail?id=1K5A2E3upOGTF6eOGgIFYISAqMDHnd59q&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1K5A2E3upOGTF6eOGgIFYISAqMDHnd59q/view)

![ctx128 with BOS/EOS emb192 layers4 bs16 lr0.0005](https://drive.google.com/thumbnail?id=1euYRjGug89VKchAo7CYDqoCiT_U2yN6I&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1euYRjGug89VKchAo7CYDqoCiT_U2yN6I/view)

### 6.3 ctx64, weight decay, epoch 실험

![ctx64 ep10 run A](https://drive.google.com/thumbnail?id=1708xktwoqIfm1H82Zzi8CLowuWiTuWm-&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1708xktwoqIfm1H82Zzi8CLowuWiTuWm-/view)

![ctx64 ep10 run B](https://drive.google.com/thumbnail?id=1s8pfHv0VwVmPGq1Ke22xu_jxq_h1WOa1&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1s8pfHv0VwVmPGq1Ke22xu_jxq_h1WOa1/view)

![ctx64 wd0.0 ep15](https://drive.google.com/thumbnail?id=1HyCgRUmjmW12CwCqqg82j4TGQatYGTzr&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1HyCgRUmjmW12CwCqqg82j4TGQatYGTzr/view)

![ctx64 wd0.01 ep15](https://drive.google.com/thumbnail?id=1SEHA9zj51JnKLLVsHzeb1VC4FwzsvBMX&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1SEHA9zj51JnKLLVsHzeb1VC4FwzsvBMX/view)

![ctx64 wd0.02 ep15](https://drive.google.com/thumbnail?id=1gNtMh75F3xoentufR8rfro0zLiJfUVpW&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1gNtMh75F3xoentufR8rfro0zLiJfUVpW/view)

### 6.4 lr와 모델 크기 보조 실험

![lr0.01 ep15 emb192](https://drive.google.com/thumbnail?id=1T-w2tDciVZeiNE7wMj3gPyZiNT9mSMAu&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1T-w2tDciVZeiNE7wMj3gPyZiNT9mSMAu/view)

![lr0.002 ep15 emb192](https://drive.google.com/thumbnail?id=1g5vSgQXbRWrL0bPPgSoubQb0e1N8Hk1A&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1g5vSgQXbRWrL0bPPgSoubQb0e1N8Hk1A/view)

![lr0.01 ep15 emb64](https://drive.google.com/thumbnail?id=1k-Wgux8dPwpF3PBaS5XKWIMGFFPvu5vJ&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1k-Wgux8dPwpF3PBaS5XKWIMGFFPvu5vJ/view)

### 6.5 최종 후보와 resume 실험

![BESTBASE lr0.01 ep20](https://drive.google.com/thumbnail?id=1oK2BhpUNOVSGrj__BnV-WyEObCCboG9X&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1oK2BhpUNOVSGrj__BnV-WyEObCCboG9X/view)

![fresh lr0.01 ep30](https://drive.google.com/thumbnail?id=1IvfKMNqO4XqAZniIH5sNaL_yPolmbRZI&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1IvfKMNqO4XqAZniIH5sNaL_yPolmbRZI/view)

![BESTPERFORMANCE resume ep20 to ep30 lr0.003](https://drive.google.com/thumbnail?id=1Br0rVQUqcxSpp3UKSZYQhM6b0pqhQfZX&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1Br0rVQUqcxSpp3UKSZYQhM6b0pqhQfZX/view)

![resume ep30 to ep35 lr0.001](https://drive.google.com/thumbnail?id=1mlRyd94MuZYCXkTZc8jcz52cQfg8DmST&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1mlRyd94MuZYCXkTZc8jcz52cQfg8DmST/view)

![resume ep35 to ep40 lr0.0005](https://drive.google.com/thumbnail?id=1HmyX1RUVB3_4G9uPC8X4fTst11wO0YIN&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1HmyX1RUVB3_4G9uPC8X4fTst11wO0YIN/view)

### 6.6 Scheduler와 gradient clipping 실험

![gradient clipping only ep20](https://drive.google.com/thumbnail?id=1DG5CPiy6zMRaQTlr2vc7hH47_HEYz2up&sz=w1200)

[원본 그래프 열기](https://drive.google.com/file/d/1DG5CPiy6zMRaQTlr2vc7hH47_HEYz2up/view)

## 7. 결론

최종 결론은 단순하다.

1. 모델을 크게 하는 것보다 데이터 경계 처리와 학습률 전략이 더 중요했다.
2. line-level BOS/EOS가 큰 폭의 loss 개선을 만들었다.
3. `lr=0.01`은 초중반에 좋지만 오래 유지하면 validation loss가 불안정해졌다.
4. 가장 효과적인 전략은 좋은 checkpoint에서 lr을 낮춰 이어 학습하는 것이었다.
5. warmup/cosine/gradient clipping은 이 설정에서는 최종 성능을 개선하지 못했다.
6. 따라서 감성 분류 fine-tuning에는 `BESTPERFORMANCE` checkpoint를 사용하는 것이 합리적이다.
