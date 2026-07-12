# Local-to-Global Temporal Multi-task Learning

이 프로젝트는 SNU AI Challenge의 4-frame temporal ordering 문제를 Qwen2-VL + LoRA로 학습하는 실험 노트북 모음입니다.

현재 메인 실험 노트북은 다음 파일입니다.

```text
local_to_global_multitask_pipeline.ipynb
```

목표는 이미 학습된 Pairwise best adapter를 초기값으로 사용해, 국소적인 두 프레임 선후관계 능력을 유지하면서 시작 프레임, 마지막 프레임, 전체 순서 판단 능력을 함께 추가 학습하는 것입니다.

## 핵심 아이디어

기존 pairwise 모델은 두 이미지 간 선후관계는 잘 배울 수 있지만, 최종 제출은 4개 이미지 전체 순서입니다. 단순히 6개 pairwise 결과를 조합하면 시작/끝 프레임을 틀리거나 전역 순서가 흔들릴 수 있습니다.

그래서 하나의 모델에 네 가지 태스크를 함께 학습합니다.

| task_type | 입력 | 출력 |
| --- | --- | --- |
| `pairwise` | Caption + 이미지 2장 | `1` 또는 `2` |
| `first` | Caption + 이미지 4장 | 시작 이미지 번호 `1~4` |
| `last` | Caption + 이미지 4장 | 마지막 이미지 번호 `1~4` |
| `order` | Caption + 이미지 4장 | 전체 순서, 예: `[3, 4, 1, 2]` |

추론 제출에서는 보조 태스크를 따로 호출하지 않고, `order` 태스크 한 번만 호출합니다.

```text
Caption + 이미지 4장 -> Full-order 1회 생성 -> submission
```

## 데이터 로딩

기존 브랜치의 Colab 노트북과 같은 경로를 사용합니다.

```python
ZIP_PATH = "/content/drive/MyDrive/SNU_AI_Challenge/snuaichallenge.zip"
DATA_DIR = "/content/snuaichallenge_data"
```

압축 해제 후 구조:

```text
/content/snuaichallenge_data/
  train.csv
  test.csv
  train/
  test/
```

CSV 컬럼은 다음을 사용합니다.

```text
Id
Sentence
Input_1
Input_2
Input_3
Input_4
Answer
```

`Answer`는 각 입력 이미지의 temporal rank입니다. 예를 들어:

```python
Answer = [3, 4, 1, 2]
```

이면 `Input_3 -> Input_4 -> Input_1 -> Input_2` 순서입니다.

## 시작 체크포인트

멀티태스크 학습은 Pairwise best adapter에서 시작합니다.

```python
PAIRWISE_BEST_ADAPTER_DIR = "/content/drive/MyDrive/SNU_AI_Challenge/qwen2vl_pairwise_v1/runs/20260712_014247_A_basic_full_1epoch/best_exact_adapter"
```

이 경로에는 반드시 `adapter_config.json`이 있어야 합니다. 없으면 새 모델로 몰래 시작하지 않고 에러로 멈춥니다.

이 학습은 `resume_from_checkpoint`가 아닙니다.

```text
Base Qwen2-VL
+ Pairwise best LoRA adapter weights
-> optimizer/scheduler 새로 시작
```

## 노트북 구성

```text
1) Install dependencies, then restart runtime once
2) Setup + data unzip
3) Data split + local-to-global multitask record generation
4) Prompt builders, dataset, collator
5) Model loading from Pairwise best adapter + task-loss Trainer
6) Train local-to-global multitask adapter
7) Validation helpers for pairwise / first / last / full-order
8) Evaluate checkpoints and choose best
9) Save selected best adapter
10) Single-call full-order test inference + submission
```

Colab에서는 1번 셀 실행 후 런타임이 재시작됩니다. 재시작 후 1번 셀을 다시 실행하고 2번부터 끝까지 실행하면 됩니다.

## 학습 설정

기본값은 전체 train split 1 epoch입니다.

```python
TRAIN_ROWS = None
VALID_ROWS = None
MAX_TRAIN_STEPS = None
```

빠른 테스트만 하고 싶으면 다음처럼 줄이면 됩니다.

```python
TRAIN_ROWS = 2000
VALID_ROWS = 300
MAX_TRAIN_STEPS = 500
```

태스크 샘플링 목표 비율:

```python
TASK_RATIOS = {
    "pairwise": 0.40,
    "first": 0.15,
    "last": 0.15,
    "order": 0.30,
}
```

`PRESERVE_ALL_PAIRWISE = True`이면 학습 subset 안의 모든 원본 샘플에 대해 6개 pairwise 관계가 최소 1번씩 포함됩니다. 그 위에 first/last/order 샘플을 oversampling해서 목표 비율에 맞춥니다.

## Loss

별도 head나 custom architecture는 만들지 않습니다. 네 태스크 모두 causal language modeling loss를 사용합니다.

중요한 구현 사항:

- prompt 영역은 `labels = -100`
- 정답 토큰에만 loss 적용
- padding token loss 제외
- task별 loss 로깅

현재 first/last를 더 강하게 학습하도록 loss weight를 둡니다.

```python
TASK_LOSS_WEIGHTS = {
    "pairwise": 1.0,
    "first": 1.0,
    "last": 1.0,
    "order": 1.0,
}
```

학습 로그에는 raw loss와 weighted loss가 함께 찍힙니다.

```text
train_pairwise_loss
train_first_loss
train_last_loss
train_order_loss
train_first_weighted_loss
train_last_weighted_loss
...
```

## 평가 지표

체크포인트마다 네 태스크를 따로 평가합니다.

Pairwise:

```text
pairwise_accuracy
```

First / Last:

```text
first_accuracy
last_accuracy
```

Full-order:

```text
valid_output_rate
position_accuracy
order_pair_accuracy
exact_match_accuracy
first_accuracy_from_order
last_accuracy_from_order
first_and_last_both_correct
exact_match_given_correct_boundaries
exact_match_given_all_pair_relations
```

오류 유형 진단:

```text
boundary_error_rate
relation_error_rate
assembly_error_rate
```

해석:

- `boundary_error_rate`: full-order 출력에서 시작 또는 마지막 프레임을 틀린 비율
- `relation_error_rate`: full-order 출력이 만드는 6개 pair 관계 중 GT와 다른 관계가 있는 비율
- `assembly_error_rate`: boundary와 pair 관계가 맞는데 exact가 틀린 경우를 잡기 위한 진단값

단, valid permutation에서 6개 pair 관계가 모두 맞으면 전체 순서는 수학적으로 하나로 결정됩니다. 따라서 실제로는 `assembly_error_rate`가 거의 0이어야 정상입니다.

## 체크포인트 선택

baseline으로 원래 Pairwise best adapter를 평가합니다. 이후 멀티태스크 checkpoint들은 다음 기준으로 선택합니다.

```text
1. baseline pairwise_accuracy 대비 하락이 -2%p 이내
2. exact_match_accuracy가 가장 높음
3. first_accuracy, last_accuracy, valid_output_rate, pairwise_accuracy 순으로 tie-break
```

선택된 adapter는 아래에 저장됩니다.

```text
runs/{RUN_ID}/lgt_multitask/best_lgt_adapter/
```

## 출력 파일

실행 결과는 Drive에 저장됩니다.

```python
LGT_ROOT = "/content/drive/MyDrive/SNU_AI_Challenge/qwen2vl_lgt_multitask_v1"
```

주요 파일:

```text
runs/{RUN_ID}/lgt_multitask/run_config.json
runs/{RUN_ID}/lgt_multitask/eval/baseline_summary.csv
runs/{RUN_ID}/lgt_multitask/eval/checkpoint_summary.csv
runs/{RUN_ID}/lgt_multitask/eval/*_pairwise_predictions.csv
runs/{RUN_ID}/lgt_multitask/eval/*_first_predictions.csv
runs/{RUN_ID}/lgt_multitask/eval/*_last_predictions.csv
runs/{RUN_ID}/lgt_multitask/eval/*_order_predictions.csv
runs/{RUN_ID}/lgt_multitask/best_lgt_adapter/
runs/{RUN_ID}/lgt_multitask/submission_lgt_multitask.csv
runs/{RUN_ID}/lgt_multitask/submission_lgt_multitask_debug.csv
```

## 추론

제출 추론은 `order` 태스크 한 번만 호출합니다.

```text
Caption + Image 1~4 -> [3, 4, 1, 2]
```

Pairwise, first, last는 학습 시 내부 표현을 강화하기 위한 보조 태스크입니다. 제출 추론에서는 사용하지 않습니다.

이 방식의 장점:

- 모델 호출 1회
- 6개 pairwise 질의 불필요
- 별도 first/last/order 모델 없음
- 별도 그래프 조합 없음
- 하나의 Qwen2-VL + 하나의 LoRA adapter만 사용

## 모순 처리

현재 제출 경로에서는 first/last/pairwise를 따로 추론해 조합하지 않으므로, 보조 태스크 간 모순을 후처리로 해결하지 않습니다.

대신 학습 후 평가에서 full-order 출력 하나를 분해해 다음을 확인합니다.

```text
시작/끝을 틀렸는가?
6개 pair 관계를 틀렸는가?
출력 형식이 invalid인가?
```

만약 나중에 first/last/pairwise/order를 모두 따로 추론해 조합하고 싶다면, 24개 permutation 후보에 대해 다음과 같은 점수를 줄 수 있습니다.

```text
score(candidate)
= order_score(candidate)
+ alpha * pair_score(candidate)
+ beta * first_score(candidate[0])
+ gamma * last_score(candidate[-1])
```

하지만 현재 노트북은 속도와 단순성을 위해 단일 full-order 호출 방식을 기본으로 합니다.

## 실행 순서

기본 실행:

```text
1 -> 런타임 재시작 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10
```

중간에 학습된 checkpoint로 직접 실험하고 싶으면 6번 학습 중 저장되는 checkpoint를 로드해 별도 inference 셀을 만들어 확인할 수 있습니다. 그래도 기본 노트북은 1 epoch 끝까지 학습하도록 설정되어 있습니다.

## 완료 조건

이 실험은 다음을 확인하기 위한 것입니다.

- Pairwise best adapter에서 멀티태스크 추가 학습 가능
- pairwise / first / last / order 데이터 자동 생성
- task 비율 조절 가능
- task별 loss와 validation 지표 기록
- full-order exact match 개선 여부 확인
- pairwise accuracy 큰 하락 여부 확인
- 최종 제출은 단일 checkpoint, 단일 full-order 호출로 동작
