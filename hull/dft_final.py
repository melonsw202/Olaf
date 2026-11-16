"""9번 준비: 학습 데이터(train/val/test.traj)에서 config별 DFT 최종(마지막 이온 스텝) 에너지 추출.
데이터는 vasprun.xml의 모든 이온 스텝이므로 config별 step 최댓값 frame = DFT 이완 최종 구조.
출력: dft_final.csv, dft_final_structures.traj (최종 구조), dft_initial_structures.traj (source 폴더의 POSCAR)
사용: python dft_final.py <data 폴더> <Ni_doped_NMO 루트> <출력 폴더>"""
import os, sys, csv, collections
import numpy as np
from ase.io import read, write

data, root, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
flag_remag = set(open(f"{root}/remag/order.txt").read().split())
flag_charge = set(open(f"{root}/mlp_model_train/analysis/flagged_chargestate.txt").read().split())

last, split_of, nfr = {}, {}, collections.Counter()
for s in ("train", "val", "test"):
    for a in read(f"{data}/{s}.traj", ":"):
        c = a.info["config"]
        nfr[c] += 1
        split_of[c] = s
        if c not in last or a.info["step"] > last[c].info["step"]:
            last[c] = a

rows, finals, inits = [], [], []
for c in sorted(last, key=lambda c: (last[c].info["x"], c)):
    a = last[c]
    src = f"{root}/{a.info['source']}"
    ini = read(f"{src}/POSCAR")
    # vasprun 이온 스텝 수와 비교해 마지막 frame이 실제 최종인지 확인 (OSZICAR의 F= 줄 수)
    try:
        n_ionic = sum("F=" in l for l in open(f"{src}/OSZICAR"))
    except OSError:
        n_ionic = -1
    f = a.get_forces()
    nna = int((a.numbers == 11).sum())
    rows.append(dict(config=c, x=a.info["x"], n_na=nna, n_atoms=len(a), split=split_of[c],
                     e_dft=a.get_potential_energy(), fmax_final=float(np.linalg.norm(f, axis=1).max()),
                     step_final=a.info["step"], n_ionic=n_ionic, n_frames=nfr[c],
                     flag_remag=c in flag_remag, flag_chargestate=c in flag_charge,
                     source=a.info["source"]))
    a.info["config"] = c; ini.info = dict(config=c, x=a.info["x"])
    finals.append(a); inits.append(ini)

with open(f"{out}/dft_final.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
write(f"{out}/dft_final_structures.traj", finals)
write(f"{out}/dft_initial_structures.traj", inits)
r = rows
print("configs", len(r), "| last frame == final ionic step:", sum(x["step_final"] == x["n_ionic"] - 1 for x in r),
      "| fmax_final>0.05:", sum(x["fmax_final"] > 0.05 for x in r),
      "| flag_remag:", sum(x["flag_remag"] for x in r), "| flag_chargestate:", sum(x["flag_chargestate"] for x in r))
print("configs per n_na:", dict(sorted(collections.Counter(x["n_na"] for x in r).items())))
