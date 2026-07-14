# Pairwise / First / Last Only Experiment

이 브랜치는 `7-refactor-order`의 LGT 데이터 세팅과 평가 구조를 기반으로, `order` 태스크를 제거하고 `pairwise / first / last` 세 태스크만 추가 학습하는 실험을 포함한다.

주요 노트북:

```text
pair_first_last_train.ipynb
pair_first_last_eval_infer.ipynb
```

## 목적

기존 full validation 결과에서 direct order generation보다 `pairwise + first + last` 확률을 결합한 structured decoding이 더 높은 성능을 보였다. 따라서 이번 실험은 `order` 생성 태스크를 학습에서 제거하고, 최종 순서를 항상 24개 permutation scoring으로 복원한다.

## 초기 체크포인트

3-task 학습은 weighted order-refine 결과 중 `checkpoint-3000`에서 시작한다.

```python
INITIAL_ADAPTER_DIR = (
    "/content/drive/MyDrive/SNU_AI_Challenge/"
    "qwen2vl_lgt_order_refine_v1/runs/"
    "20260714_005408/lgt_order_refine/checkpoint-3000"
)
```

평가에서는 비교 기준으로 다음도 함께 본다.

- 기존 LGT `checkpoint-3500`
- weighted order-refine `checkpoint-3000`
- 새 3-task checkpoint들

## 학습 태스크

사용 태스크:

```text
pairwise
first
last
```

제거 태스크:

```text
order
adjacent
```

비율:

```python
TASK_RATIOS = {
    "pairwise": 0.40,
    "first": 0.30,
    "last": 0.30,
}
```

Loss weight:

```python
TASK_LOSS_WEIGHTS = {
    "pairwise": 1.0,
    "first": 1.0,
    "last": 1.0,
}
```

학습 설정:

```python
LEARNING_RATE = 5e-6
MAX_TRAIN_STEPS = 1200
SAVE_STEPS = 300
```

## 저장 위치

```text
/content/drive/MyDrive/SNU_AI_Challenge/qwen2vl_pair_first_last_only_v1/runs/{RUN_ID}/pair_first_last_only/
```

생성 checkpoint:

```text
checkpoint-300
checkpoint-600
checkpoint-900
checkpoint-1200
final_adapter
```

## 추론 방식

Direct order generation은 사용하지 않는다. 각 샘플에서 다음 확률만 추출한다.

- `P(i before j)`
- `P_first(i)`
- `P_last(i)`

그 뒤 가능한 24개 순열을 모두 점수화한다.

```python
score(order) =
    alpha * pair_score(order)
  + beta  * log P_first(order[0])
  + gamma * log P_last(order[-1])
```

Grid:

```python
ALPHAS = [0.5, 1.0, 1.5, 2.0]
BETAS = [0.5, 1.0, 1.5, 2.0]
GAMMAS = [0.5, 1.0, 1.5, 2.0]
```

## 평가 산출물

```text
eval/all_checkpoint_metrics_quick.csv
eval/all_checkpoint_metrics_full300_{PROMPT_VERSION}.csv
eval/*_{PROMPT_VERSION}_task_probability_cache.json
eval/*_decoding_weight_search.csv
eval/*_checkpoint_predictions.csv
eval/*_endpoint_agreement_analysis.csv
best_adapter/best_config.json
submission_pair_first_last.csv
```

추가 분석:

- Direct first와 pairwise-derived first 일치 여부
- Direct last와 pairwise-derived last 일치 여부
- task first/last 동시 정답률
- decoded first/last 동시 정답률
- task first/last 동시 정답일 때 exact match
- decoded first/last 동시 정답일 때 exact match

## 평가 분리

평가 cache에는 prompt version을 포함한다.

```python
PROMPT_VERSION = "chrono_explicit_v1"
```

Full validation은 같은 300개에서 가중치 선택과 성능 보고를 동시에 하지 않도록 분리한다.

```text
tuning 150: alpha/beta/gamma 선택
holdout 150: 선택된 가중치로 checkpoint 비교
```

Full 평가에는 항상 다음 baseline이 포함된다.

- 기존 LGT `checkpoint-3500`
- weighted order-refine `checkpoint-3000`
- quick 결과 상위 새 checkpoint들

## 실행 순서

1. `pair_first_last_train.ipynb` 실행
2. 3-task checkpoint 저장 확인
3. `pair_first_last_eval_infer.ipynb` 실행
4. quick validation으로 checkpoint 선별
5. full300 validation 평가
6. best checkpoint 저장
7. test submission 생성
