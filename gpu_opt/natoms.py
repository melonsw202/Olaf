import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
L="/proj/home/ibs/tdqh/hgpark/calc/battery/Na_cathode/Ni_doped_NMO/lsw/mlp_data_set/"
fr=read(L+"train.traj", index=":")
n=np.array([len(a) for a in fr])
print("structures",len(n),"natoms min/median/max",n.min(),np.median(n),n.max())
print("natoms counts:",dict(zip(*np.unique(n,return_counts=True))))
rng=np.random.default_rng(0); idx=rng.choice(len(fr),200,replace=False)
e=np.array([len(neighbor_list("i",fr[i],7.0)) for i in idx])
print("edges(sample200) min/median/max",e.min(),int(np.median(e)),e.max(),"edges/atom",round(float((e/n[idx]).mean()),1))
for cap in [15000,30000,60000,120000]:
    print(f"cap {cap}: graphs per batch ~ {cap/np.median(e):.2f}")
