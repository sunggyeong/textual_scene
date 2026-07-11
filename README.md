# SNU AI Challenge - Frame Order Prediction

이 프로젝트는 4장의 섞인 비디오 프레임을 보고 각 이미지의 실제 시간 순위를 예측하는 Colab용 파이프라인입니다. 기본 모델은 `Qwen/Qwen2-VL-2B-Instruct`이고, 4bit quantization + LoRA 방식으로 학습합니다.

## 파일 구성

- `colab_pipeline.ipynb`: 권장 실행 노트북입니다. setup, dataset, train, eval, inference가 5개 셀로 정리되어 있습니다.
- `pre.ipynb`: 데이터 압축 해제와 기본 경로 확인용 기존 노트북입니다.
- `train.ipynb`: 학습용 기존 노트북입니다.
- `eval.ipynb`: 검증 평가용 기존 노트북입니다.
- `inference.ipynb`: 제출 파일 생성용 기존 노트북입니다.

새로 실행할 때는 `colab_pipeline.ipynb`만 사용하는 것을 권장합니다.

## 데이터 위치

Colab 기준으로 아래 zip 파일이 Google Drive에 있어야 합니다.

```python
ZIP_PATH = "/content/drive/MyDrive/SNU_AI_Challenge/snuaichallenge.zip"
```

압축을 풀면 아래 구조를 기대합니다.

```text
/content/snuaichallenge_data/
  train.csv
  test.csv
  train/
  test/
```

학습 결과와 checkpoint는 기본적으로 아래에 저장됩니다.

```python
OUTPUT_DIR = "/content/drive/MyDrive/SNU_AI_Challenge/qwen2vl_multitask_v1"
```

제출 파일은 아래에 생성됩니다.

```python
SUBMIT_PATH = "/content/outputs/submission.csv"
```

## 권장 실행 방법

Colab에서 `colab_pipeline.ipynb`를 열고 위에서 아래로 실행합니다.

### 처음부터 학습 + 평가 + 추론

아래 5개 셀을 순서대로 실행합니다.

```text
1) Install dependencies, then restart runtime once
2) Setup + data unzip
3) Dataset, collator, prediction helpers
4) Train
5) Eval + best checkpoint by exact True
6) Inference + submission with best checkpoint loaded above
```

`3) Train` 셀은 학습 후 `OUTPUT_DIR`에 LoRA adapter와 checkpoint를 저장합니다.

### 이미 학습된 가중치로 평가 + 추론만 실행

이미 `OUTPUT_DIR`에 학습된 adapter/checkpoint가 있다면 `3) Train` 셀은 건너뜁니다.

실행 순서:

```text
1) Install dependencies, then restart runtime once
2) Setup + data unzip
3) Dataset, collator, prediction helpers
5) Eval + best checkpoint by exact True
6) Inference + submission with best checkpoint loaded above
```

단, train 셀을 건너뛰면 `processor`와 `bnb_config`가 아직 없을 수 있으므로 4번 셀 전에 아래 미니 셀을 한 번 실행합니다.

```python
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

4번 셀은 `OUTPUT_DIR`와 `OUTPUT_DIR/checkpoint-*` 안의 adapter들을 validation exact-match 기준으로 평가하고, 가장 많이 맞힌 checkpoint를 `model`에 다시 로드합니다. 이후 5번 셀을 실행하면 선택된 best checkpoint로 `submission.csv`를 생성합니다.

## 학습 방식

학습 데이터셋은 multi-task 형식입니다.

```python
TRAIN_TASKS = ("ORDER", "PAIRWISE", "RANK")
EVAL_TASKS = ("ORDER",)
```

- `ORDER`: 4개 이미지 각각의 실제 시간 순위를 리스트로 예측합니다. 예: `[2, 1, 4, 3]`
- `PAIRWISE`: 두 이미지 중 더 먼저 나온 프레임을 `A` 또는 `B`로 예측합니다.
- `RANK`: 특정 Input 이미지의 실제 시간 순위를 `1`부터 `4` 중 하나로 예측합니다.

평가와 제출은 대회 제출 형식에 맞게 `ORDER`만 사용합니다.

제출 답안 형식은 다음 의미입니다.

```text
[Input_1의 실제 순위, Input_2의 실제 순위, Input_3의 실제 순위, Input_4의 실제 순위]
```

즉 `[1, 2, 3, 4]`는 `Input_1`이 첫 번째 프레임, `Input_2`가 두 번째 프레임이라는 뜻입니다.

## 평가 항목

`colab_pipeline.ipynb`의 4번 셀은 다음을 확인합니다.

- validation exact-match accuracy
- 맞힌 개수
- 파싱 실패 개수
- 항상 `[1, 2, 3, 4]`를 내는 baseline accuracy
- train/validation `Id` 중복 여부
- 동일 프레임 묶음 중복 여부
- `No_ordering` 컬럼이 있을 경우 그룹별 position accuracy와 exact-match accuracy
- 저장된 checkpoint별 exact-match accuracy

checkpoint별 평가는 시간이 오래 걸릴 수 있습니다. 빠르게 확인하려면 1번 셀의 값을 조정합니다.

```python
BEST_CHECKPOINT_EVAL_LIMIT = 50
```

전체 validation으로 평가하려면 기본값처럼 `None`을 사용합니다.

```python
BEST_CHECKPOINT_EVAL_LIMIT = None
```

## 주요 설정

`colab_pipeline.ipynb`의 1번 셀에서 주로 바꿀 수 있는 값입니다.

```python
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
VALID_RATIO = 0.1
SMOKE_TEST = False
TRAIN_TASKS = ("ORDER", "PAIRWISE", "RANK")
EVAL_TASKS = ("ORDER",)
BEST_CHECKPOINT_EVAL_LIMIT = None
```

빠른 smoke test를 하고 싶으면:

```python
SMOKE_TEST = True
```

## 주의사항

- Colab GPU 런타임 사용을 전제로 합니다.
- `bitsandbytes`, `peft`, `transformers`, `qwen-vl-utils`가 설치됩니다.
- batch size는 collator 구현상 1을 전제로 합니다.
- `processor`는 원본 모델인 `MODEL_ID`에서 로드하는 것이 안전합니다.
- `OUTPUT_DIR`에는 LoRA adapter/checkpoint가 저장됩니다.
