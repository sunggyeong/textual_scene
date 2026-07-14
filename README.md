# Adjacent Structured Multi-task Pipeline

이 브랜치는 기존 LGT multi-task 모델에 `Adjacent-next` 태스크를 추가하고, 최종 순서 예측을 direct-order 생성이 아니라 structured decoding으로 수행하기 위한 실험 노트북을 포함한다.

핵심 노트북:

```text
adjacent_structured_pipeline.ipynb
```

## 목표

기존 모델은 다음 태스크를 함께 학습했다.

- Pairwise
- First
- Last
- Direct Order

하지만 제출 추론에서 direct-order 생성만 사용할 경우, 보조 태스크에서 배운 Pairwise / First / Last 정보를 최종 순서 결정에 직접 활용하지 못한다.

이 노트북은 여기에 `Adjacent-next` 태스크를 추가한다. 이후 validation/test 추론에서는 모델이 생성한 단일 order 문자열만 쓰지 않고, 다음 확률들을 조합해 24개 가능한 순열을 전수 점수화한다.

- 첫 이미지 확률
- 마지막 이미지 확률
- 두 이미지 간 선후관계 확률
- 바로 다음 이미지 확률
- 마지막 이미지의 `next = 0` 확률

## 초기 체크포인트

새 학습은 기존 학습을 optimizer까지 이어서 재개하지 않는다. 아래 adapter 가중치만 초기값으로 불러오고 optimizer/scheduler는 새로 시작한다.

```python
INITIAL_ADAPTER_DIR = (
    "/content/drive/MyDrive/SNU_AI_Challenge/"
    "qwen2vl_lgt_multitask_v1/runs/"
    "20260712_234828/lgt_multitask/checkpoint-3500"
)
```

출력 루트:

```python
OUTPUT_ROOT = (
    "/content/drive/MyDrive/SNU_AI_Challenge/"
    "qwen2vl_adjacent_structured_v1"
)
```

## 학습 태스크

학습에는 다음 5개 태스크를 섞어서 사용한다.

| 태스크 | 역할 |
| --- | --- |
| `pairwise` | 두 이미지 중 어느 쪽이 먼저인지 학습 |
| `first` | 전체 이야기의 시작 프레임 학습 |
| `last` | 전체 이야기의 마지막 프레임 학습 |
| `adjacent` | 특정 이미지 바로 다음에 오는 이미지 학습 |
| `order` | 4개 이미지 전체 순서 학습 |

기본 비율:

```python
TASK_RATIOS = {
    "pairwise": 0.30,
    "first": 0.15,
    "last": 0.15,
    "adjacent": 0.25,
    "order": 0.15,
}
```

Loss weight는 모두 `1.0`으로 둔다. 첫 실험에서는 데이터 비율만 바꾸고 loss weight는 바꾸지 않아야 원인 해석이 쉽다.

## Adjacent-next 태스크

정답 순서가 다음과 같다면:

```text
[3, 4, 1, 2]
```

생성되는 adjacent 샘플은 다음과 같다.

```text
next(3) = 4
next(4) = 1
next(1) = 2
next(2) = 0
```

여기서 `0`은 더 이상 뒤에 오는 이미지가 없다는 뜻이다.

프롬프트 형식:

```text
Caption: ...
Images:
Image 1
Image 2
Image 3
Image 4
Which image comes immediately after Image 3?
Return exactly one digit: 0 if there is no following image, otherwise 1, 2, 3, or 4.
```

마지막 이미지의 `next=0`을 학습하고 평가에 반영해야 마지막 프레임 판단이 adjacent 확률과도 연결된다.

## 주요 설정

```python
LEARNING_RATE = 1e-5
MAX_TRAIN_STEPS = 1200
SAVE_STEPS = 300
LOGGING_STEPS = 20

QUICK_EVAL_ROWS = 50
FULL_EVAL_ROWS = 300
CALIBRATION_EVAL_ROWS = 150
HOLDOUT_EVAL_ROWS = 150
TOP_K_FULL_EVAL = 2
```

생성되는 주요 checkpoint:

```text
checkpoint-300
checkpoint-600
checkpoint-900
checkpoint-1200
```

초기 `checkpoint-3500`도 structured baseline으로 함께 평가한다.

## Structured Decoding

각 validation/test 샘플에 대해 모델에서 다음 확률을 얻는다.

- `first_probs[i]`: `i`번 이미지가 첫 이미지일 확률
- `last_probs[i]`: `i`번 이미지가 마지막 이미지일 확률
- `pair_probs[a>b]`: `a`번 이미지가 `b`번 이미지보다 먼저일 확률
- `adjacent_probs[a>b]`: `a`번 이미지 바로 다음이 `b`번일 확률
- `adjacent_probs[a>0]`: `a`번 이미지가 마지막일 확률

그 뒤 가능한 24개 순열을 모두 점수화한다.

```python
score(order) =
    alpha * log(P(first = order[0]))
  + beta  * boundary_last_score
  + gamma * mean(log(P(order[i] before order[j])))
  + delta * mean(log(P(order[i+1] after order[i])))
```

`boundary_last_score`는 다음 두 정보를 같이 사용한다.

```text
P(last = order[-1])
P(next(order[-1]) = 0)
```

Grid search 후보:

```python
ALPHAS = [0.5, 1.0, 1.5]
BETAS = [0.5, 1.0, 1.5]
GAMMAS = [1.0]
DELTAS = [0.5, 1.0, 1.5]
```

## 평가 흐름

1. 학습 후 checkpoint 목록 수집
2. 초기 `checkpoint-3500` 포함
3. 각 checkpoint를 quick validation 50개로 평가
4. quick structured exact match 기준 상위 2개 선택
5. validation 300개를 calibration 150 / holdout 150으로 분리
6. calibration에서 structured weight grid search
7. holdout에서 선택된 weight로 최종 비교
8. best checkpoint를 `best_structured_adapter/`에 저장

같은 300개에서 weight를 고르고 같은 데이터로 성능을 보고하지 않도록 calibration/holdout을 분리한다.

## Ablation

다음 네 조건을 저장한다.

| 이름 | 사용 정보 |
| --- | --- |
| `A_pairwise_only` | Pairwise only |
| `B_pairwise_first_last` | Pairwise + First/Last |
| `C_pairwise_adjacent` | Pairwise + Adjacent |
| `D_all` | Pairwise + First/Last + Adjacent |

각 ablation 모드는 독립적으로 grid search를 수행한다.

## 저장 파일

실행 결과는 다음 위치에 저장된다.

```text
/content/drive/MyDrive/SNU_AI_Challenge/qwen2vl_adjacent_structured_v1/runs/{RUN_ID}/adjacent_multitask/
```

주요 파일:

| 파일/폴더 | 설명 |
| --- | --- |
| `checkpoint-*` | 300 step 간격 LoRA checkpoint |
| `final_adapter/` | 학습 종료 시점 adapter |
| `eval/*_probs.json` | checkpoint별 모델 확률 캐시 |
| `eval/quick_structured_summary.csv` | quick validation 결과 |
| `eval/full_structured_holdout_summary.csv` | calibration/holdout 기반 최종 비교 |
| `best_structured_adapter/` | 최종 선택된 adapter |
| `best_structured_adapter/best_structured_config.json` | best checkpoint와 decoding weight |
| `best_structured_adapter/structured_weight_search.csv` | grid search 전체 결과 |
| `best_structured_adapter/structured_ablation_summary.csv` | ablation 결과 |
| `submission_adjacent_structured_best.csv` | test 제출 CSV |

## 중단 복구

노트북은 결과 파일이 이미 있으면 가능한 부분을 건너뛴다.

- checkpoint가 이미 있으면 재사용
- 확률 JSON이 있으면 재추론 생략
- quick/full summary가 있어도 새 checkpoint가 추가되면 그 checkpoint만 추가 평가

단, 학습 자체를 완전 재개하려면 `trainer.train(resume_from_checkpoint=...)` 형태의 별도 수정이 필요하다. 현재 노트북은 `checkpoint-3500` adapter를 초기값으로 새 학습을 시작하도록 설계되어 있다.

## 제출 형식

Structured decoding 결과는 chronological order 형태로 나온다.

예:

```text
[3, 4, 1, 2]
```

제출 CSV에서는 기존 train label과 맞게 rank-per-input 형태로 변환한다.

```python
def sequence_to_answer(order):
    answer = [0] * 4
    for rank, image_number in enumerate(order, start=1):
        answer[int(image_number) - 1] = rank
    return answer
```

예를 들어 chronological sequence가 `[2, 3, 4, 1]`이면 다음처럼 변환된다.

```text
[4, 1, 2, 3]
```

일반적으로 chronological sequence와 rank-per-input은 다르므로 반드시 변환을 거쳐야 한다.

## 실행 순서

Colab에서 `adjacent_structured_pipeline.ipynb`를 위에서 아래로 실행한다.

권장 흐름:

1. Drive mount 및 패키지 설치
2. 첫 셀이 런타임을 재시작하면, 재시작 후 첫 셀을 다시 실행하고 다음 셀로 진행
3. 두 번째 셀에서 Drive mount, 데이터 압축 해제 및 기본 경로 확인
4. task record 생성 확인
5. 모델 로드
6. 1200 step 학습
7. quick evaluation
8. calibration/holdout full evaluation
9. best adapter 저장
10. test submission CSV 생성
