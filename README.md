# Olaf

Na cathode (Ni-doped NMO) BAM-torch 파인튜닝 프로젝트.

- `gpu_opt/` : 파인튜닝 GPU 효율 최적화 (설정 비교, 추천 input.json / run.sh)
- `eval/` : FT 모델 test set 평가 (E/F/S MAE)
- `hull/` : DFT vs FT convex hull, voltage profile 비교
- `active/iterN/` : 불일치 조성 추가 구조 생성 → DFT → 재파인튜닝 반복

대용량 파일(.pkl, .traj, VASP 출력 등)은 .gitignore로 제외하고 클러스터에만 보관.
