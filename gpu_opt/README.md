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
