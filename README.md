# LGT Order Refine Pipeline

이 브랜치는 기존 LGT multi-task 모델의 `checkpoint-3500`을 초기값으로 사용해, Adjacent 없이 `order / pairwise / first / last` 네 태스크만 다시 학습하는 Colab 파이프라인을 포함한다.

주요 노트북:

```text
lgt_order_refine_train.ipynb
lgt_order_refine_eval_infer.ipynb
```

기존 통합 노트북과 Adjacent 실험 노트북도 남아 있지만, 현재 요청 기준의 권장 실행 파일은 위 두 개이다.

## 목적

Adjacent 태스크를 추가한 실험에서는 adjacent accuracy가 일부 개선되더라도 전체 order exact match가 기존 `checkpoint-3500`보다 좋아지지 않았다.

따라서 이번 실험은 Adjacent를 제거하고, 기존 LGT 태스크만 유지한다.

- Order
- Pairwise
- First
- Last

대신 학습 샘플 비율을 조정해 Order와 First/Last를 더 명확하게 강화한다.

## 초기 체크포인트

새 학습은 기존 checkpoint를 optimizer까지 이어서 재개하지 않는다. 아래 LoRA adapter 가중치만 불러오고 optimizer/scheduler는 새로 시작한다.

```python
INITIAL_ADAPTER_DIR = (
    "/content/drive/MyDrive/SNU_AI_Challenge/"
    "qwen2vl_lgt_multitask_v1/runs/"
    "20260712_234828/lgt_multitask/checkpoint-3500"
)
```

## Base Model 캐시

Hugging Face 큰 shard 다운로드가 `0B/s`에서 멈추는 경우를 피하기 위해, 노트북은 기본적으로 ModelScope로 base model을 다운로드한 뒤 Drive에 캐시한다.

```python
MODEL_REPO_ID = "Qwen/Qwen2-VL-2B-Instruct"
USE_MODELSCOPE_BASE_MODEL = True
DRIVE_MODEL_DIR = (
    "/content/drive/MyDrive/SNU_AI_Challenge/"
    "model_cache/Qwen2-VL-2B-Instruct"
)
```

처음 실행할 때 `DRIVE_MODEL_DIR/config.json`이 없으면 ModelScope에서 다운로드하고, 이후에는 로컬 폴더에서만 로드한다.

```python
MODEL_ID = ensure_base_model_path()
MODEL_LOCAL_FILES_ONLY = os.path.isdir(MODEL_ID)
```

집이나 안정적인 네트워크에서 Hugging Face를 직접 쓰고 싶다면 설정 셀에서 다음처럼 바꿀 수 있다.

```python
USE_MODELSCOPE_BASE_MODEL = False
```

## 출력 폴더

기존 LGT 결과를 덮어쓰지 않도록 새 폴더에 저장한다.

```python
OUTPUT_ROOT = (
    "/content/drive/MyDrive/SNU_AI_Challenge/"
    "qwen2vl_lgt_order_refine_v1"
)
```

실제 run별 저장 위치:

```text
/content/drive/MyDrive/SNU_AI_Challenge/qwen2vl_lgt_order_refine_v1/runs/{RUN_ID}/lgt_order_refine/
```

## 학습 설정

기본 태스크 비율:

```python
TASK_RATIOS = {
    "order": 0.40,
    "pairwise": 0.20,
    "first": 0.20,
    "last": 0.20,
}
```

Loss weight는 모두 `1.0`으로 둔다.

```python
TASK_LOSS_WEIGHTS = {
    "order": 1.0,
    "pairwise": 1.0,
    "first": 1.0,
    "last": 1.0,
}
```

학습 파라미터:

```python
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 1
MAX_TRAIN_STEPS = -1
SAVE_STEPS = 100
LOGGING_STEPS = 20
```

`MAX_TRAIN_STEPS = -1`이므로 step 수로 강제 종료하지 않고 `NUM_TRAIN_EPOCHS` 기준으로 학습한다. checkpoint는 100 step마다 계속 저장된다.

생성되는 checkpoint 예:

```text
checkpoint-100
checkpoint-200
...
final_adapter
```

## 실행 순서

### 1. 학습

Colab에서 `lgt_order_refine_train.ipynb`를 실행한다.

1. 첫 셀 실행
2. dependency 설치 후 런타임이 재시작되면 첫 셀 재실행
3. 두 번째 셀에서 Drive mount 및 기본 경로 확인
4. 데이터 unzip 및 train/validation split
5. `order / pairwise / first / last` task record 생성
6. `checkpoint-3500` adapter 로드
7. 1 epoch 학습
8. 100 step마다 checkpoint 저장

### 2. 평가 및 추론

학습이 어느 정도 진행되거나 끝난 뒤 `lgt_order_refine_eval_infer.ipynb`를 실행한다.

기본값은 `qwen2vl_lgt_order_refine_v1/runs/` 아래 가장 최신 run을 자동 선택한다. 특정 run을 평가하려면 설정 셀에서 다음 값을 지정한다.

```python
REFINE_RUN_ID = "실제_RUN_ID"
```

평가 노트북은 다음을 수행한다.

1. 저장된 checkpoint 목록 수집
2. `BASELINE_ADAPTER_DIR`의 기존 `checkpoint-3500` baseline 포함
3. checkpoint별 quick validation 평가
4. 상위 checkpoint full validation 평가
5. best adapter 저장
6. test submission 생성

평가 노트북의 `BASELINE_ADAPTER_DIR`는 학습용이 아니라 비교 기준이다. 학습 비율은 직접 평가 계산에 쓰지 않고, 학습 run의 `run_config.json`에서 읽어 `best_config.json` 기록용으로만 사용한다.

## 평가 방식

노트북은 두 가지 추론 방식을 비교한다.

### 1. Direct Order Generation

Order prompt를 입력하고 모델이 직접 생성한 순서 리스트를 사용한다.

예:

```text
[3, 4, 1, 2]
```

### 2. Pairwise + First + Last Probability Decoding

각 샘플에 대해 다음 확률을 구한다.

- `P(i before j)`
- `P_first(i)`
- `P_last(i)`

그 다음 24개 가능한 순열을 모두 점수화한다.

```python
score(order) =
    alpha * pairwise_score(order)
  + beta  * first_score(order[0])
  + gamma * last_score(order[-1])
```

Grid search 범위:

```python
ALPHAS = [0.5, 1.0, 1.5, 2.0]
BETAS = [0.5, 1.0, 1.5, 2.0]
GAMMAS = [0.5, 1.0, 1.5, 2.0]
```

## 저장 파일

주요 산출물:

```text
eval/all_checkpoint_metrics_quick.csv
eval/all_checkpoint_metrics_full.csv
eval/*_task_probability_cache.json
eval/*_decoding_weight_search.csv
eval/*_direct_predictions.csv
eval/*_structured_predictions.csv
best_adapter/
best_adapter/best_config.json
best_adapter/all_checkpoint_metrics.csv
submission_lgt_order_refine.csv
```

## 제출 형식

모델 내부 예측은 chronological sequence이다.

예:

```text
[2, 3, 4, 1]
```

제출 CSV는 train label과 같은 rank-per-input 형식으로 변환한다.

```python
def sequence_to_answer(order):
    answer = [0] * 4
    for rank, image_number in enumerate(order, start=1):
        answer[int(image_number) - 1] = rank
    return answer
```

예를 들어 `[2, 3, 4, 1]`은 다음처럼 변환된다.

```text
[4, 1, 2, 3]
```

## 참고

`lgt_order_refine_pipeline.ipynb`는 학습과 평가를 한 파일에 합친 이전 버전이다. 현재처럼 학습을 오래 돌리고 checkpoint 검사는 따로 하려면 `lgt_order_refine_train.ipynb`와 `lgt_order_refine_eval_infer.ipynb`를 사용한다.

`adjacent_structured_pipeline.ipynb`는 이전 Adjacent 실험용으로 남아 있다.
