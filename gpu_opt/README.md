# gpu_opt: 파인튜닝 GPU 효율 최적화

클러스터 작업 위치: `olaf:/proj/home/ibs/tdqh/hgpark/calc/battery/Na_cathode/Ni_doped_NMO/lsw/calc_opt/`
(원본 `calc/`는 수정하지 않음. 모델 `.pkl`은 클러스터에만 보관)

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `template/` | 원본 `calc/`의 `input.json`, `run.sh`, `enr_avg_per_element.txt` 복사본 (변경 없음) |
| `00_baseline/` | 기준값 측정: 원본 설정 + `nepoch: 3`, 모델 경로는 복사본 |
| `setup_baseline.sh` | `00_baseline/`을 생성하고 job을 제출한 스크립트 |

`00_baseline/input.json`에서 원본과 다른 부분:
- `NN.nepoch`: 200 → 3
- `NN.fname_pkl`, `predict.model`, `enr_avg_per_element`: `calc_opt/00_baseline/` 안의 복사본을 가리키도록 변경

`00_baseline/run.sh`에서 원본과 다른 부분:
- 작업 이름 `opt_00_baseline`, 제한 시간 3일 → 1시간
- `gpu_mem.log`(nvidia-smi 15초 간격 기록)와 `WALL_SEC` 출력 추가

## 결과: 00_baseline (job 42922311, 2026-11-06)

| 항목 | 값 |
|---|---|
| 파티션 | mig-3g.40gb (H100 MIG), `-c 8` |
| 데이터 | train 13,654개, val 1,427개 구조 |
| epoch 119 (1번째) | 14:47:45 → 15:09:08, 약 21.4분 (첫 데이터 전처리 포함) |
| epoch 120 (2번째) | 15:09:08 → 15:29:53, **약 20.8분 (1,245초)** ← 기준값 |
| 초당 처리 구조 수 | **약 11개/초** |
| GPU 메모리 | 측정 실패 (아래 참고) |
| 종료 상태 | TIMEOUT (01:00:25). epoch 121은 학습 부분이 거의 끝난 상태에서 중단 |

200 epoch로 환산하면 약 70시간이 걸립니다. 원본 제한 시간(72시간)에 거의 닿는 수준입니다.

### loss 이상 여부: 이상 없음
- epoch이 일반 학습과 DeNS 학습으로 번갈아 진행되면서, 그 epoch에 계산하지 않는 항목이 `nan`으로 표시됩니다.
  - 일반 epoch(119, 121): `LOSS_DN = nan`
  - DeNS epoch(120): `LOSS_S = nan` (흔든 구조는 응력 정답이 없음)
- 전체 train loss는 0.0149 → 0.0114로 줄었고, validation loss도 정상 값입니다.

### 참고
- `gpu_mem.log`: MIG 환경이라 `nvidia-smi`에 같은 GPU를 쓰는 다른 사용자의 프로세스만 보였습니다. 이 job의 메모리 값으로 쓸 수 없습니다. 다음 테스트부터는 `torch.cuda.max_memory_allocated`로 측정할 예정입니다.
- `nepoch: 3`이면 테스트 하나가 1시간을 넘깁니다. 다음 테스트부터는 로그 각 줄에 시각을 찍고 `nepoch: 1`로 비교할 예정입니다.

## 1차: 01_baseline_re vs 02_no_gradckpt (2026-11-06)

공통: `nepoch: 1`, mig-3g.40gb, `-c 8`, 원본 모델(epoch 118) 복사본에서 시작.
`tools/mem_wrap.py`로 `torch.cuda.max_memory_allocated`를 기록하고, `run.sh`에서 출력 각 줄에 시각을 붙임.

| 테스트 | 변경 | job | 학습 시작 → epoch 로그 | 전체 실행 | 최대 GPU 메모리 (alloc / reserved) | loss |
|---|---|---|---|---|---|---|
| 01_baseline_re | 없음 | 42925762 | 16:55:48 → 17:16:40 (20.9분) | 1,404초 | 7.01 / 14.16 GiB | 00_baseline epoch 119와 완전히 같음 |
| 02_no_gradckpt | `grad_checkpoint: false` | 42925763 | 16:55:48 → 17:16:36 (20.8분) | 1,404초 | 7.02 / 14.16 GiB | 위와 완전히 같음 |

### 결론
- **`grad_checkpoint`는 효과가 없는 옵션입니다.** `bam_torch`와 설치된 `env` 전체에서 이 키를 읽는 코드가 없습니다. `examples/bam-mp-dens/README.md`에도 "기록 보존용으로 남긴 설정이며 이 RACE 클래스에서는 gradient checkpointing을 켜지 않는다"고 적혀 있습니다. 시간, 메모리, loss가 모두 같은 것과도 일치합니다.
- 기준값을 다시 재도 epoch당 약 21분으로 재현됩니다.
- **GPU 메모리는 40GB 중 약 7GB(예약 포함 14GB)만 쓰고 있습니다.** 배치를 키울 여유가 큽니다.

### 원인 분석: 실제 배치 크기가 4보다 훨씬 작음
`BucketedDataLoader`(`bam_torch/utils/sampler.py`)는 배치에 구조를 최대 `nbatch`개까지 넣다가, 간선 수 합이 `max_edges_per_batch` 이상이 되면 거기서 멈춥니다.

학습 데이터를 직접 확인한 결과 (`natoms.py`):
- 구조당 원자 수 54~72개 (중앙값 62)
- cutoff 7 Å 기준 구조당 간선 수 6,112~9,658개 (중앙값 7,741개, 200개 표본)

그래서 `max_edges_per_batch: 15000`이면 구조 2개만 넣어도 대부분 15,000을 넘습니다. **실제 배치는 대부분 구조 1개**이고, `nbatch: 4`에는 거의 도달하지 않습니다. GPU 사용률이 낮은 주된 원인으로 보입니다.

참고: `bucket_reference`는 코드에서 `'edges'`로 고정되어 있어 설정값이 쓰이지 않습니다. `log_interval`은 epoch 단위로 로그와 validation을 건너뛰는 옵션이라 1 epoch 테스트로는 측정할 수 없습니다.

## 2차: 배치 크기 키우기 (2026-11-12)

공통: 1차와 같음 (`nepoch: 1`, mig-3g.40gb, `-c 8`, 원본 모델 epoch 118 복사본에서 시작).

| 테스트 | 변경 | job | 학습 시간 (1 epoch) | 최대 GPU 메모리 (alloc / reserved) | 결과 |
|---|---|---|---|---|---|
| 01_baseline_re (참고) | 없음 (실제 배치 약 1개) | 42925762 | 20.9분 | 7.01 / 14.16 GiB | 완료 |
| 03_edges32k | `max_edges_per_batch` 32000 (실제 배치 약 4개) | 43558460 | **12.0분** (15:43:00 → 15:55:01) | 14.50 / 28.21 GiB | 완료, **약 1.7배 빠름** |
| 04_nb8_edges64k | `nbatch` 8, `max_edges_per_batch` 64000 (실제 배치 약 8개) | 43558461 | - | 28.25 / 38.77 GiB | **학습 시작 직후 OOM** (39.5 GiB 중 712 MiB 추가 할당 실패) |

### 1 epoch 후 loss 비교 (같은 시작점, 같은 learning rate 1e-4)

| 테스트 | train LOSS | valid LOSS | valid HUBER_E | valid HUBER_F | valid HUBER_S |
|---|---|---|---|---|---|
| 시작점 (epoch 118 checkpoint) | 0.0125 | 0.0003 | - | - | - |
| 01_baseline_re | 0.01489 | **0.000112** | 0.00948 | 0.02415 | 0.00126 |
| 03_edges32k | 0.01813 | **0.000244** | 0.01677 | 0.02589 | 0.00195 |

- 03은 epoch 시간이 약 1.7배 짧지만, 1 epoch 후 valid loss는 기준의 약 2.2배입니다.
- 배치가 약 4배가 되면서 epoch당 가중치 업데이트 횟수가 약 1/4로 줄어든 영향으로 보입니다 (추정: 약 13,600회 → 약 3,400회).
- 따라서 epoch 시간만으로는 판단할 수 없습니다. 같은 learning rate에서는 epoch당 학습 진행이 느려집니다. 여러 epoch에 걸친 학습 거동 비교와 learning rate 조정이 필요합니다 (지침 6번).

## 3차: 배치 크기 × learning rate, 3 epoch 학습 거동 (2026-11-12)

공통: 지금 모델(epoch 118)에서 이어서 3 epoch, mig-3g.40gb, `-c 8`.
restart 시 checkpoint의 optimizer 상태가 `input.json`의 learning rate를 덮어쓰므로, 테스트 폴더의 모델 복사본 안 `opt_state.param_groups[].lr`(와 scheduler `_last_lr`)만 바꿨습니다 (`setup_round3.sh`). 로그 LR 칸에서 적용을 확인했습니다.

| 테스트 | 배치 (`nbatch` / `max_edges`) | lr | 학습 시간 (3 epoch) | epoch당 | 최대 메모리 alloc / reserved |
|---|---|---|---|---|---|
| 05_bs1_lr1e-4 | 4 / 15000 (실제 약 1) | 1e-4 | 64.0분 | 21.3분 | 7.04 / 20.63 GiB |
| 06_bs4_lr1e-4 | 4 / 32000 (실제 약 4) | 1e-4 | 35.2분 | 11.7분 | 14.52 / 28.61 GiB |
| 07_bs4_lr2e-4 | 4 / 32000 | 2e-4 | 35.4분 | 11.8분 | 14.52 / 28.21 GiB |
| 08_bs4_lr4e-4 | 4 / 32000 | 4e-4 | 36.0분 | 12.0분 | 14.52 / 28.61 GiB |

학습 시간: 로그 헤더 출력 시각 → 마지막 epoch 로그 시각.

### validation MAE (epoch 119 → 120 → 121)

| 테스트 | 에너지 MAE (meV/atom) | 힘 MAE (eV/Å) | 응력 MAE |
|---|---|---|---|
| 05 배치1 lr1e-4 | 9.48 → 9.29 → **9.38** | 0.0241 → 0.0296 → **0.0222** | 0.00126 → 0.00256 → **0.00125** |
| 06 배치4 lr1e-4 | 16.77 → 9.57 → **9.48** | 0.0259 → 0.0299 → **0.0231** | 0.00195 → 0.00216 → **0.00132** |
| 07 배치4 lr2e-4 | 16.39 → 9.29 → **9.48** | 0.0242 → 0.0285 → **0.0218** | 0.00143 → 0.00233 → **0.00129** |
| 08 배치4 lr4e-4 | 15.93 → 9.42 → **9.46** | 0.0231 → 0.0271 → **0.0207** | 0.00125 → 0.00256 → **0.00123** |

(로그의 valid `HUBER_E/F/S` 칸. `log_config`상 `loss_e/f/s` = MAE. 에너지는 ×1000 해서 meV/atom)

### 결론
- 배치 약 4개는 첫 epoch에서만 뒤처지고, 2번째 epoch부터 기준과 같은 수준이 됩니다. 3 epoch 뒤 에너지 MAE는 네 테스트 모두 9.4~9.5 meV/atom.
- learning rate를 키울수록 힘 MAE가 조금씩 좋아졌고(0.0231 → 0.0218 → 0.0207), 4e-4에서도 loss가 튀거나 발산하지 않았습니다. 08은 3 epoch 뒤 힘·응력 MAE가 기준(05)보다도 낮습니다.
- 같은 3 epoch에 학습 시간은 64분 → 36분 (약 44% 단축). 200 epoch 환산 약 71시간 → 약 40시간.
- 주의: 3 epoch, seed 1개 결과라 힘 MAE의 몇 % 차이는 잡음일 수 있습니다. epoch 120(DeNS epoch)은 모든 테스트에서 힘·응력 MAE가 높게 나오는 공통 패턴입니다.

### 참고: valid LOSS = 에너지 항만
네 테스트 모두 valid `LOSS` 값이 `LOSS_E_H`와 같습니다. checkpoint 선택과 `ReduceLROnPlateau`가 valid loss를 쓰면 힘·응력은 반영되지 않을 수 있습니다. 의도된 동작인지 확인이 필요합니다.
