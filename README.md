# Pilot C: 7B Multitask + Bidirectional Pair + Conditional Order

이 브랜치는 Qwen2-VL 7B base model에서 새 LoRA adapter를 학습하고, 조건부 순서 디코딩까지 평가하는 Pilot C 실험용 노트북을 포함한다.

구현 파일은 두 개로 나뉜다.

```text
train_qwen2vl_7b_multitask_bipair_conditional.ipynb
eval_qwen2vl_7b_multitask_bipair_conditional.ipynb
```

## 실험 목적

기존 order / first / last / pairwise 멀티태스크 학습에 다음 요소를 추가한다.

- pairwise 입력을 정방향과 역방향으로 모두 학습한다.
- first 또는 last가 고정된 조건부 순서 예측 태스크를 학습한다.
- first와 last가 모두 고정된 상태에서 middle frame 순서를 예측하는 태스크를 학습한다.
- 추론 시 order, endpoint, bidirectional pairwise 신호를 단계적으로 결합하는 conditional sequential decoder를 평가한다.

최종 목표는 기존 단순 weighted permutation decoder보다 `conditional_sequential_exact`가 실제 validation에서 개선되는지 확인하는 것이다.

## 데이터와 모델 경로

기본 경로는 `7-refactor-order` 계열 노트북과 동일하게 맞췄다.

```text
/content/drive/MyDrive/SNU_AI_Challenge/snuaichallenge.zip
/content/snuaichallenge_data/train.csv
/content/snuaichallenge_data/test.csv
/content/snuaichallenge_data/train/
/content/snuaichallenge_data/test/
```

base model cache:

```text
/content/drive/MyDrive/SNU_AI_Challenge/model_cache/Qwen2-VL-7B-Instruct
```

없으면 ModelScope에서 `Qwen/Qwen2-VL-7B-Instruct`를 내려받아 위 경로에 저장한다.

split은 가능하면 기존 split을 재사용한다.

```text
/content/drive/MyDrive/SNU_AI_Challenge/id_splits/qwen2vl_lgt_order_refine_20260714_003635/
```

해당 split이 없으면 `SEED = 42`, `VALID_RATIO = 0.10` 기준으로 train/validation을 새로 나눈다.

## 학습 노트북

파일:

```text
train_qwen2vl_7b_multitask_bipair_conditional.ipynb
```

학습은 기존 adapter에서 이어서 하지 않고, 7B base model에서 새 LoRA adapter를 만든다.

주요 설정:

```python
MODEL_REPO_ID = "Qwen/Qwen2-VL-7B-Instruct"
OUTPUT_ROOT = "/content/drive/MyDrive/SNU_AI_Challenge/qwen2vl_7b_multitask_bipair_conditional_v1"
SAVE_STEPS = 100
save_total_limit = None
```

출력 구조:

```text
qwen2vl_7b_multitask_bipair_conditional_v1/
└── runs/
    └── {RUN_ID}/
        └── multitask_bipair_conditional/
            ├── checkpoint-100/
            ├── checkpoint-200/
            ├── ...
            ├── final_adapter/
            ├── eval/
            └── task_distribution.json
```

### 학습 태스크 비율

| Task | Ratio |
|---|---:|
| Full order | 30% |
| Bidirectional pairwise | 25% |
| First | 10% |
| Last | 10% |
| Fixed-first conditional order | 10% |
| Fixed-last conditional order | 10% |
| Fixed-endpoints middle order | 5% |

### 태스크 설명

`order`: 전체 4개 frame의 chronological order를 출력한다.

```text
3 2 1 4
```

`pairwise`: 두 이미지를 A/B로 제시하고, 더 먼저 일어나는 이미지를 `A` 또는 `B`로 출력한다. unordered pair마다 정방향과 역방향 입력을 모두 만든다.

`first`: 전체 4개 frame 중 첫 장면 번호만 출력한다.

`last`: 전체 4개 frame 중 마지막 장면 번호만 출력한다.

`fixed_first`: gold first frame을 조건으로 주고, 나머지 3개 frame 순서만 출력한다.

`fixed_last`: gold last frame을 조건으로 주고, 나머지 3개 frame 순서만 출력한다.

`fixed_endpoints`: gold first와 gold last를 모두 조건으로 주고, 가운데 2개 frame 순서만 출력한다.


## 재개 학습 노트북

기존 run을 이어서 학습하려면 아래 파일을 사용한다.

```text
resume_train_qwen2vl_7b_multitask_bipair_conditional.ipynb
```

사용 방법:

1. 노트북의 `# 2) Setup` 셀에서 `RESUME_RUN_ID`를 기존 run id로 지정한다.
2. 셀을 순서대로 실행한다.
3. 노트북은 해당 run의 `multitask_bipair_conditional/checkpoint-*` 중 가장 큰 step을 찾아 `resume_from_checkpoint`로 이어서 학습한다.
4. 재개 학습이 끝나면 기존 평가 노트북에서 같은 run id를 `PILOT_RUN_ID`로 지정해 평가한다.

예:

```python
RESUME_RUN_ID = "20260721_012345"
```

## 평가 및 추론 노트북

파일:

```text
eval_qwen2vl_7b_multitask_bipair_conditional.ipynb
```

이 노트북은 다음을 수행한다.

1. `runs/{RUN_ID}/multitask_bipair_conditional/` 아래 checkpoint 자동 탐색
2. 모든 checkpoint quick validation
3. 상위 checkpoint 정밀 validation
4. best adapter 저장
5. test submission 생성

특정 run을 평가하려면 eval 노트북의 설정에서 값을 지정한다.

```python
PILOT_RUN_ID = "실제_RUN_ID"
```

`None`이면 가장 최신 run을 사용한다.

## Conditional Sequential Decoder

평가 노트북의 최종 decoder는 다음 신호를 사용한다.

- full order 24개 후보 teacher-forcing score
- first / last head probability
- bidirectional pairwise probability
- fixed-first 또는 fixed-last conditional order score
- fixed-endpoints middle order score

### Bidirectional pairwise

각 unordered pair `(i, j)`에 대해 두 방향을 모두 평가한다.

```python
p_forward = P(A | input=[i, j])
p_reverse = P(A | input=[j, i])
```

`i before j` 방향으로 변환한 뒤 logit 평균을 사용한다.

```python
combined_logit = 0.5 * (logit(p_forward) - logit(p_reverse))
combined_prob = sigmoid(combined_logit)
```

swap inconsistency도 함께 저장한다.

```python
swap_inconsistency = abs(p_forward + p_reverse - 1.0)
```

### 디코딩 흐름

1. 24개 full order 후보를 모두 score한다.
2. order 분포에서 first/last marginal을 계산한다.
3. order marginal, endpoint head, pairwise endpoint score를 결합한다.
4. first와 last 중 top1-top2 margin이 큰 endpoint를 먼저 고정한다.
5. 고정된 endpoint 조건으로 conditional order 후보를 다시 score한다.
6. 반대 endpoint를 고른다.
7. first/last가 모두 고정된 상태에서 middle order를 score한다.
8. middle conditional score와 bidirectional pair score를 결합해 최종 순서를 만든다.

## 주요 평가 지표

quick/full validation에서 저장하는 핵심 지표:

```text
direct_exact
direct_position
conditional_sequential_exact
conditional_sequential_first
conditional_sequential_last
conditional_sequential_both_endpoints
conditional_sequential_position
conditional_sequential_relative_pair
bidirectional_pair_accuracy
mean_swap_inconsistency
```

best checkpoint 선택 우선순위:

1. `conditional_sequential_exact`
2. `direct_exact`
3. `conditional_sequential_both_endpoints`
4. `bidirectional_pair_accuracy`
5. `conditional_sequential_relative_pair`

## 산출물

평가 결과:

```text
eval/checkpoint_metrics_quick.csv
eval/checkpoint_metrics_full.csv
eval/sample_predictions/{checkpoint}_quick*.json
eval/sample_predictions/{checkpoint}_full*.json
```

best adapter:

```text
best_adapter/
├── adapter_config.json
├── adapter_model.safetensors
├── best_checkpoint.json
├── checkpoint_metrics_full.csv
└── submission.csv
```

submission:

```text
submission_pilot_c.csv
```

## 실행 순서

1. Colab GPU 런타임에서 학습 노트북을 연다.
2. dependency 셀을 실행한다. 설치 후 런타임이 재시작되면 다시 첫 셀부터 실행한다.
3. 학습을 실행한다. checkpoint는 100 step마다 저장된다.
4. 학습이 충분히 진행되면 평가 노트북을 연다.
5. 필요하면 `PILOT_RUN_ID`를 지정한다.
6. quick eval과 full eval을 실행한다.
7. best adapter와 submission을 확인한다.

## 주의

- 7B 모델이므로 Colab GPU 메모리가 부족할 수 있다.
- conditional sequential evaluation은 full order 24개, conditional 6개, middle 2개 후보를 teacher forcing으로 평가하므로 일반 direct inference보다 느리다.
- train 중 quick eval callback은 매우 작은 subset만 사용한다. 최종 판단은 eval 노트북의 full validation 결과를 기준으로 한다.
