"""8번 test set 평가 후처리. 사용: python analyze_eval.py <eval 폴더>
- 공식 MultiheadEvaluator는 checkpoint 안의 enr_avg_per_element(사전학습 MP 값)를 더함.
  FT 학습은 input.json의 enr_avg_per_element 파일 값을 썼으므로, FT 모델은 그 차이를 보정해야 학습과 같은 에너지가 됨.
- 출력: 원자당 에너지 MAE/RMSE, 힘 MAE/RMSE, 응력 MAE, Na 수에 대한 선형 기준 보정 후 에너지 MAE, Na 수별 에너지 MAE
"""
import ast, json, sys, collections
import numpy as np, torch
from ase.io import read

d = sys.argv[1]
cfg = json.load(open(f"{d}/input.json"))
ckpt = torch.load(cfg["predict"]["model"], map_location="cpu", weights_only=False)
uniq = ckpt["uniq_element"]
enr_ckpt = ckpt["enr_avg_per_element"]
enr_file, _ = ast.literal_eval(open(cfg["enr_avg_per_element"]).read())
tv = torch.load(f"{d}/test_values.pkl", weights_only=False)
traj = read(cfg["predict"]["fname_traj"], ":")
assert len(traj) == len(tv["energy"]), (len(traj), len(tv["energy"]))

is_ft = "lsw/calc" in cfg["predict"]["model"]
res = collections.defaultdict(list)
for a, ep, ex, fx, fy, fz in zip(traj, tv["energy"], tv["exact_energy"], tv["force_x"], tv["force_y"], tv["force_z"]):
    sp = [uniq[int(z)] for z in a.numbers]
    e = float(ep)
    if is_ft:   # 학습 때 쓴 기준 에너지로 교체
        e += sum(enr_file[s] - enr_ckpt[s] for s in sp)
    n = len(a)
    res["n"].append(n); res["nna"].append(int((a.numbers == 11).sum()))
    res["dE"].append((e - a.get_potential_energy()) / n)
    f_pred = np.stack([fx.numpy(), fy.numpy(), fz.numpy()], 1)
    res["dF"].append((f_pred - a.get_forces()).ravel())
dE = np.array(res["dE"]); nna = np.array(res["nna"]); n = np.array(res["n"])
dF = np.concatenate(res["dF"])
# 기준 보정: 총에너지 오차 = a*N_Na + b (Mn/Ni/O 수는 고정) 를 최소제곱으로 빼기
A = np.stack([nna, np.ones_like(nna)], 1).astype(float)
coef, *_ = np.linalg.lstsq(A, dE * n, rcond=None)
dE_c = (dE * n - A @ coef) / n
out = {
    "n_frames": len(traj),
    "energy_mae_meV_atom": 1000 * np.abs(dE).mean(),
    "energy_rmse_meV_atom": 1000 * np.sqrt((dE**2).mean()),
    "energy_mean_err_meV_atom": 1000 * dE.mean(),
    "energy_mae_refcorr_meV_atom": 1000 * np.abs(dE_c).mean(),
    "refcorr_coef_eV": {"per_Na": coef[0], "const": coef[1]},
    "force_mae_eV_A": np.abs(dF).mean(),
    "force_rmse_eV_A": np.sqrt((dF**2).mean()),
    "by_nNa_energy_mae_meV_atom": {int(k): 1000 * np.abs(dE[nna == k]).mean() for k in sorted(set(nna))},
}
for line in open(f"{d}/predict.out"):
    if line.startswith("MAE(S)"):
        out["stress_mae_evaluator"] = float(line.split()[1])
json.dump(out, open(f"{d}/metrics.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
