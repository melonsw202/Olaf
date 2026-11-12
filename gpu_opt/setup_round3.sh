# 3차: 배치 크기 x learning rate 학습 거동 비교 (지금 모델 epoch 118에서 이어서)
# 동시 실행 2개 제한: [05 -> 08], [06 -> 07] 두 줄로 순차 실행 (afterany 의존성)
set -e
L=/proj/home/ibs/tdqh/hgpark/calc/battery/Na_cathode/Ni_doped_NMO/lsw
O=$L/calc_opt
PY=$L/env/bin/python
[ -f $O/tools/mem_wrap.py ] || { echo "mem_wrap.py 없음"; exit 1; }

# $1=폴더 $2=nbatch $3=max_edges $4=lr $5=nepoch $6=제한시간 $7=sbatch 추가옵션
make_test () {
  T=$O/$1
  [ -e "$T" ] && { echo "EXISTS: $T (abort)"; exit 1; }
  mkdir -p $T
  cp -p $L/calc/enr_avg_per_element.txt $T/
  # 모델 복사본의 optimizer/scheduler learning rate만 변경 (원본은 읽기만 함)
  # restart 시 optimizer.load_state_dict()가 checkpoint의 lr로 input.json 값을 덮어쓰기 때문
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
  $PY - "$L/calc/input.json" "$T/input.json" "$T" "$2" "$3" "$4" "$5" <<'EOF'
import json, sys
src, dst, d, nb, me, lr, ne = sys.argv[1:]
c = json.load(open(src))
c["nbatch"] = int(nb)
c["max_edges_per_batch"] = int(me)
c["NN"]["learning_rate"] = float(lr)
c["scheduler"]["lr_init"] = float(lr)
c["NN"]["nepoch"] = int(ne)
c["NN"]["fname_pkl"] = f"{d}/model_ft.pkl"
c["predict"]["model"] = f"{d}/model_ft.pkl"
c["enr_avg_per_element"] = f"{d}/enr_avg_per_element.txt"
json.dump(c, open(dst, "w"), indent=1)
EOF
  sed -e "s/__NAME__/opt_$1/" -e "s/__TIME__/$6/" > $T/run.sh <<'EOF'
#!/bin/bash
#SBATCH -J __NAME__
#SBATCH -p mig-3g.40gb
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
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
  echo "--- $1: diff vs 원본"
  diff $L/calc/input.json $T/input.json | grep -E '^[<>]' | grep -v -E 'fname_pkl|"model"|enr_avg' || true
  JID=$(cd $T && sbatch --parsable $7 run.sh)
  echo "SUBMITTED $1 $JID"
}

make_test 05_bs1_lr1e-4 4 15000 1e-4 3 01:20:00 ""
J05=$JID
make_test 06_bs4_lr1e-4 4 32000 1e-4 3 00:50:00 ""
J06=$JID
make_test 07_bs4_lr2e-4 4 32000 2e-4 3 00:50:00 "--dependency=afterany:$J06"
make_test 08_bs4_lr4e-4 4 32000 4e-4 3 00:50:00 "--dependency=afterany:$J05"
squeue -u hgpark
