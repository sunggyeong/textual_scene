# Pairwise Qwen2-VL Pipeline

4개 프레임의 전체 순서를 한 번에 맞히는 ORDER 방식 대신, 두 프레임씩 비교하는 PAIRWISE 모델을 학습하고 평가하는 Colab 노트북입니다.

모델은 각 pair에 대해 다음 중 하나를 예측합니다.

```text
1: 첫 번째 이미지가 더 이른 프레임
2: 두 번째 이미지가 더 이른 프레임
```

한 샘플의 6개 pair 결과를 모아 가능한 24개 전체 순서 중 pair probability 합이 가장 높은 순서를 최종 답으로 복원합니다.

## 노트북

사용 노트북은 하나입니다.

```text
pairwise_pipeline.ipynb
```

셀 구성:

```text
1) Install dependencies, then restart runtime once
2) Setup + data unzip
3) Pairwise dataset, collator, scoring helpers
4) Train pairwise LoRA                    # 수동 단일 variant용
5) Eval checkpoints                       # 수동 단일 variant용
6) Inference + submission                 # 수동 단일 variant용
7) Generate event/frame-description caches
8) Run A/B/C/D ablation end-to-end
```

기본값은 `RUN_ABLATION_ALL = True`입니다. 전체 셀을 위에서 아래로 실행하면 4~6번 수동 셀은 자동으로 건너뛰고, 7~8번 셀이 캐시 생성과 A/B/C/D 실험을 처리합니다.

## 데이터

기본 Colab 경로:

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

## 저장 위치

PAIRWISE 결과는 ORDER 결과를 덮어쓰지 않도록 별도 Drive 폴더에 저장합니다.

```python
PAIRWISE_ROOT = "/content/drive/MyDrive/SNU_AI_Challenge/qwen2vl_pairwise_v1"
```

매 실행마다 timestamp run 폴더가 생깁니다.

```text
qwen2vl_pairwise_v1/runs/YYYYMMDD_HHMMSS/
```

주요 산출물:

```text
runs/{RUN_ID}/A_basic/
runs/{RUN_ID}/B_events/
runs/{RUN_ID}/C_descriptions/
runs/{RUN_ID}/D_full/
runs/{RUN_ID}/ablation_summary.csv
```

각 variant 폴더 안에는 checkpoint, eval CSV, best full validation 결과가 저장됩니다. 최종 test inference CSV는 validation 기준 best variant 폴더에만 저장됩니다.

```text
runs/{RUN_ID}/{BEST_VARIANT}/test_pair_predictions.csv
runs/{RUN_ID}/{BEST_VARIANT}/submission_pairwise_debug.csv
runs/{RUN_ID}/{BEST_VARIANT}/submission_pairwise.csv
```

## Pairwise 학습 방식

한 샘플에서 항상 6개 pair 전체를 사용합니다.

```text
(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)
```

학습 중에는 pair의 이미지 제시 순서를 seed 기반으로 일부 뒤집고, 그에 맞게 target도 뒤집습니다. Validation과 inference에서도 같은 6개 pair를 모두 사용합니다.

기본 1차 실험은 오래 걸리지 않도록 고정 subset을 사용합니다.

```python
ABLATION_TRAIN_ROWS = 300
ABLATION_VALID_ROWS = 100
MAX_TRAIN_STEPS = 300
```

A/B/C/D 모두 동일한 train/validation split, 동일한 subset, 동일한 seed, 동일한 LoRA 설정, 동일한 step 수를 사용합니다. 바뀌는 것은 prompt에 들어가는 정보뿐입니다.

전체 데이터로 본 실험을 하려면 다음처럼 바꾸면 됩니다.

```python
ABLATION_TRAIN_ROWS = None
ABLATION_VALID_ROWS = None
MAX_TRAIN_STEPS = None
```

## Prompt Variant

| Variant | 입력 정보 |
| --- | --- |
| `A_BASIC` | 이미지 2장 + Sentence |
| `B_EVENTS` | 이미지 2장 + Sentence + 사건 단계 |
| `C_DESCRIPTIONS` | 이미지 2장 + Sentence + 두 프레임의 순수 시각 설명 |
| `D_FULL` | 이미지 2장 + Sentence + 사건 단계 + 두 프레임의 순수 시각 설명 |

중요한 비교 조건:

- Train: 원본 샘플당 6개 pair 전부
- Validation: 원본 샘플당 6개 pair 전부
- 동일한 train/validation split
- 동일한 ablation subset
- 동일한 seed
- 동일한 base model과 LoRA 설정
- 동일한 optimizer step
- 동일한 이미지 순서 뒤집기 규칙

따라서 성능 차이는 prompt에 추가된 정보의 영향으로 해석할 수 있습니다.

## 캐시

B/C/D는 prompt에 넣을 보조 텍스트가 필요해서 캐시 JSON을 사용합니다.

```python
GENERATE_CACHES = True
EVENTS_CACHE_PATH = ".../cache/events_qwen2vl.json"
FRAME_DESCRIPTIONS_CACHE_PATH = ".../cache/frame_descriptions_qwen2vl.json"
```

7번 셀은 기본적으로 ablation train subset 300개와 validation subset 100개에 필요한 캐시만 생성합니다. test 캐시는 처음부터 만들지 않습니다. 8번 셀에서 A/B/C/D 평가가 끝난 뒤 best variant를 고르고, 그 best variant가 B/C/D라면 test에 필요한 캐시만 추가로 생성한 다음 inference를 실행합니다.

캐시 종류:

- event cache: Sentence를 1~6개의 시간 순 사건 단계로 분해
- frame description cache: Sentence 없이 이미지만 보고 순수 시각 정보 설명

T4에서 안전하게 돌리기 위한 기본 batch:

```python
EVENT_CACHE_BATCH_SIZE = 8
FRAME_CACHE_BATCH_SIZE = 2
```

메모리가 부족하면 `FRAME_CACHE_BATCH_SIZE = 1`로 줄이면 됩니다.

## 전체 실행

Colab에서 기본 설정 그대로 실행:

```text
1 -> 런타임 재시작 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8
```

1번 셀은 dependency 설치 후 런타임을 강제로 재시작합니다. 재시작 후 1번 셀을 다시 실행하고, 그 다음 2번부터 내려가면 됩니다.

`RUN_ABLATION_ALL = True`일 때 실제 흐름:

```text
7번: ablation train/valid subset 캐시 생성
8번: A_BASIC train -> checkpoint eval -> best full validation
8번: B_EVENTS train -> checkpoint eval -> best full validation
8번: C_DESCRIPTIONS train -> checkpoint eval -> best full validation
8번: D_FULL train -> checkpoint eval -> best full validation
8번: ablation_summary.csv 저장
8번: validation 최고 variant 선택
8번: 최고 variant에 필요한 test cache만 생성
8번: 최고 variant checkpoint reload 후 test inference
```

즉 전체 셀을 실행해 두면 A/B/C/D 비교표와 최종 submission CSV까지 생성됩니다.

## 평가

checkpoint 선택 기준은 다음 우선순위입니다.

```text
1. position_accuracy
2. pair_accuracy
3. exact_match_accuracy
```

저장되는 주요 평가 파일:

```text
eval/checkpoint_summary.csv
eval/*_pair_predictions.csv
eval/*_reconstructed_orders.csv
eval/best_full_pair_predictions.csv
eval/best_full_reconstructed_orders.csv
```

`ablation_summary.csv`에는 variant별로 다음 지표가 모입니다.

- pair_accuracy
- position_accuracy
- exact_match_accuracy
- pair_correct / pair_total
- exact_correct / order_total
- best_checkpoint

## 기존 학습 결과 평가/추론

이미 저장된 pairwise run을 평가하거나 추론하려면 2번 셀에서 자동 ablation을 끄고 기존 output 폴더를 지정합니다.

```python
RUN_ABLATION_ALL = False
EXISTING_OUTPUT_DIR = "/content/drive/MyDrive/SNU_AI_Challenge/qwen2vl_pairwise_v1/runs/YYYYMMDD_HHMMSS/A_basic"
```

그 다음 실행 순서:

```text
1 -> 2 -> 3 -> 5 -> 6
```

저장된 adapter checkpoint가 없으면 새 모델을 몰래 평가하지 않고 에러로 멈춥니다.

## 모델 설정

- Base model: `Qwen/Qwen2-VL-2B-Instruct`
- 학습 방식: 4bit quantization + LoRA
- Vision encoder freeze
- LoRA target modules: `q_proj`, `v_proj`
- target token: `"1"` 또는 `"2"`
- optimizer: `paged_adamw_8bit`
- per-device train batch size: 1
- gradient accumulation: 8
- checkpoint 저장: 100 optimizer step마다

## 결과 해석

비교는 다음처럼 보면 됩니다.

```text
A -> B 상승: Sentence 사건 분해가 도움됨
A -> C 상승: 구조화된 시각 설명이 도움됨
B/C -> D 상승: 두 정보를 함께 쓰는 것이 도움됨
```

1차 300-step subset 실험에서 좋은 variant가 나오면, 그 variant만 전체 train/validation 설정으로 늘려 본 실험을 돌리면 됩니다.
