# 4차: (1) 추천 후보 vs 기준 10 epoch 학습 곡선  (2) 추천 후보 기준 단일 변수 1 epoch 속도 비교
# 동시 실행 2개 제한: [09 -> 11 -> 12 -> 13 -> 14], [10] 두 줄 (afterany 의존성)
set -e
L=/proj/home/ibs/tdqh/hgpark/calc/battery/Na_cathode/Ni_doped_NMO/lsw
O=$L/calc_opt
PY=$L/env/bin/python
[ -f $O/tools/mem_wrap.py ] || { echo "mem_wrap.py 없음"; exit 1; }

# $1=폴더 $2=nbatch $3=max_edges $4=lr $5=nepoch $6=제한시간 $7=CPU수 $8=oeq_conv_fusion(true/false) $9=sbatch 추가옵션
make_test () {
  T=$O/$1
  [ -e "$T" ] && { echo "EXISTS: $T (abort)"; exit 1; }
  mkdir -p $T
  cp -p $L/calc/enr_avg_per_element.txt $T/
  # 모델 복사본의 optimizer/scheduler lr만 변경 (restart 시 checkpoint lr가 input.json 값을 덮어씀)
  $PY - "$L/calc/model_ft.pkl" "$T/model_ft.pkl" "$4" <<'EOF'
import sys, torch
src, dst, lr = sys.argv[1], sys.argv[2], float(sys.argv[3])
c = torch.load(src, map_location="cpu", weights_only=False)
for g in c["opt_state"]["param_groups"]:
    g["lr"] = lr
s = c.get("scheduler")
if isinstance(s, dict) and "_last_lr" in s:
    s["_last_lr"] = [lr] * len(s["_last_lr"])
torch.save(c, dst)
print("ckpt lr ->", [g["lr"] for g in torch.load(dst, map_location="cpu", weights_only=False)["opt_state"]["param_groups"]])
EOF
  $PY - "$L/calc/input.json" "$T/input.json" "$T" "$2" "$3" "$4" "$5" "$8" <<'EOF'
import json, sys
src, dst, d, nb, me, lr, ne, fu = sys.argv[1:]
c = json.load(open(src))
c["nbatch"] = int(nb)
c["max_edges_per_batch"] = int(me)
c["NN"]["learning_rate"] = float(lr)
c["scheduler"]["lr_init"] = float(lr)
c["NN"]["nepoch"] = int(ne)
c["NN"]["fname_pkl"] = f"{d}/model_ft.pkl"
c["predict"]["model"] = f"{d}/model_ft.pkl"
c["enr_avg_per_element"] = f"{d}/enr_avg_per_element.txt"
def set_fusion(x):
    if isinstance(x, dict):
        for k in x:
            if k == "oeq_conv_fusion": x[k] = (fu == "true")
            else: set_fusion(x[k])
set_fusion(c)
json.dump(c, open(dst, "w"), indent=1)
EOF
  sed -e "s/__NAME__/opt_$1/" -e "s/__TIME__/$6/" -e "s/__CPU__/$7/" > $T/run.sh <<'EOF'
#!/bin/bash
#SBATCH -J __NAME__
#SBATCH -p mig-3g.40gb
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c __CPU__
#SBATCH -t __TIME__
#SBATCH -o slurm-%j.out
L=/proj/home/ibs/tdqh/hgpark/calc/battery/Na_cathode/Ni_doped_NMO/lsw
cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
T0=$(date +%s)
$L/env/bin/python -u $L/calc_opt/tools/mem_wrap.py \
    $L/BAM-torch_DeNS/examples/bam-mp-dens/main_gpu.py input.json 2>&1 \
  | while IFS= read -r line; do printf '%s %s\n' "$(date +%T)" "$line"; done
echo "WALL_SEC $(( $(date +%s) - T0 ))"
EOF
  chmod +x $T/run.sh
  echo "--- $1: diff vs 원본 (-c $7)"
  diff $L/calc/input.json $T/input.json | grep -E '^[<>]' | grep -v -E 'fname_pkl|"model"|enr_avg' || true
  JID=$(cd $T && sbatch --parsable $9 run.sh)
  echo "SUBMITTED $1 $JID"
}

# (1) 10 epoch 학습 곡선 비교
make_test 09_bs4_lr4e-4_10ep 4 32000 4e-4 10 02:40:00 8 true ""
J=$JID
make_test 10_bs1_lr1e-4_10ep 4 15000 1e-4 10 04:30:00 8 true ""
# (2) 추천 후보(배치 4, lr 4e-4)에서 한 가지씩만 바꿔 1 epoch 속도 비교 (기준: 03_edges32k 12.0분, -c 8, fusion on)
make_test 11_fusion_off 4 32000 4e-4 1 00:40:00 8 false "--dependency=afterany:$J"; J=$JID
make_test 12_nb6_edges48k 6 48000 4e-4 1 00:40:00 8 true "--dependency=afterany:$J"; J=$JID
make_test 13_cpu4 4 32000 4e-4 1 00:40:00 4 true "--dependency=afterany:$J"; J=$JID
make_test 14_cpu16 4 32000 4e-4 1 00:40:00 16 true "--dependency=afterany:$J"
squeue -u hgpark -p mig-3g.40gb
