# eval: test set 평가 (지침 8번)

클러스터 위치: `lsw/eval/` (모델은 읽기만 함)

| 파일 | 내용 |
|---|---|
| `setup_eval.sh` | 평가 폴더 생성 + job 제출. 공식 `MultiheadEvaluator`를 그대로 실행 (`tools/run_eval.py`) |
| `analyze_eval.py` | `test_values.pkl` 후처리: 원자당 에너지 MAE, 힘 MAE, Na 수에 대한 선형 기준 보정 후 에너지 MAE |

## 평가 코드 주의점
- `MultiheadEvaluator`는 checkpoint 안의 EMA 가중치와 `enr_avg_per_element`를 씁니다.
- 학습 때 쓴 `enr_avg_per_element` 파일 값과 checkpoint 값이 다르면 에너지가 그만큼 어긋나므로 `analyze_eval.py`에서 학습 기준으로 교체합니다 (경로에 `lsw/calc`가 들어간 모델만).
- "기준 보정 MAE": 총에너지 오차를 `a·N_Na + b`로 최소제곱 맞춘 뒤 남은 오차. 원소 기준 에너지 차이(사전학습 MP 기준 vs 이 DFT 설정)를 빼고 비교하기 위한 값. test set에서 두 계수를 맞추므로 약간 낙관적입니다.

## 결과 (test.traj 1,819 frame, 23개 config)

| 모델 | 에너지 MAE (meV/atom) | 기준 보정 후 에너지 MAE | 힘 MAE (eV/Å) | 힘 RMSE |
|---|---|---|---|---|
| 사전학습 BAM-MP-DeNS | 838.5 (기준 에너지 차이) | 5.6 | 0.0646 | 0.0933 |
| 원본 `calc/model_ft.pkl` | 사전학습과 동일 (같은 가중치) | | | |
| 4차 09 테스트 모델 (파인튜닝 2 epoch에서 저장) | 8.7 | 6.1 | 0.0338 | 0.0584 |

- 원본 `calc/model_ft.pkl`은 사전학습 모델과 같은 가중치라 별도 평가 의미가 없습니다 (`gpu_opt/README.md` 참고).
- 실제 파인튜닝 모델이 생기면 같은 방법으로 다시 평가합니다.
