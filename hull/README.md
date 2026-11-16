# hull: DFT vs FT convex hull, voltage (지침 9번)

| 파일 | 내용 |
|---|---|
| `dft_final.py` | 학습 데이터(train/val/test.traj)에서 config별 DFT 최종 이온 스텝 에너지·구조 추출 → `dft_final.csv` |

## DFT 데이터 현황 (2026-11-16)
- config 237개 (Na 0~18개). 237개 모두 데이터의 마지막 frame이 OSZICAR의 마지막 이온 스텝과 일치.
- 최종 구조의 최대 힘 > 0.05 eV/Å: 6개 (EDIFFG = -0.02, 완전히 이완되지 않음)
- Na 12, 13개 조성은 config가 3개씩뿐.
- 41개 config는 선배님 분석(`mlp_model_train/analysis/flagged_chargestate.txt`, `remag/order.txt`)에서 비정상 산화 상태로 표시되어 MAGMOM을 바꿔 재계산(remag) 중. 지금 학습 데이터(`lsw/mlp_data_set`, 11/4 생성)에는 이 41개의 원래 계산이 들어 있습니다.
- Na 금속 기준(`na_ref/bcc`)은 입력만 있고 계산 결과가 없습니다. 전압 절댓값에는 필요하지만, DFT와 FT 전압의 차이에는 영향이 없습니다.
