set -e
L=/proj/home/ibs/tdqh/hgpark/calc/battery/Na_cathode/Ni_doped_NMO/lsw
O=$L/calc_opt
mkdir -p $O/tools

# --- 메모리 기록 래퍼 (학습 코드는 수정하지 않고 감싸서 실행) ---
cat > $O/tools/mem_wrap.py <<'PY'
"""main_gpu.py를 그대로 실행하면서 GPU 최대 메모리를 30초마다 출력.
사용: python mem_wrap.py <main_gpu.py> <input.json>"""
import sys, time, threading, signal, runpy, torch

def report(tag):
    if torch.cuda.is_available():
        a = torch.cuda.max_memory_allocated() / 2**30
        r = torch.cuda.max_memory_reserved() / 2**30
        print(f"[mem] {tag} {time.strftime('%H:%M:%S')} max_alloc={a:.2f}GiB max_reserved={r:.2f}GiB", flush=True)

def loop():
    while True:
        time.sleep(30)
        report("periodic")

def on_term(sig, frame):
    report("sigterm")
    sys.exit(1)

signal.signal(signal.SIGTERM, on_term)
threading.Thread(target=loop, daemon=True).start()
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    report("final")
PY

make_test () {   # $1=폴더명  $2=input.json 수정용 python 코드
  T=$O/$1
  [ -e "$T" ] && { echo "EXISTS: $T (abort)"; exit 1; }
  mkdir -p $T
  cp -p $L/calc/model_ft.pkl $L/calc/enr_avg_per_element.txt $T/
  $L/env/bin/python - "$L/calc/input.json" "$T/input.json" "$T" "$2" <<'PY'
import json, sys
src, dst, d, extra = sys.argv[1:]
c = json.load(open(src))
c["NN"]["nepoch"] = 1
c["NN"]["fname_pkl"] = f"{d}/model_ft.pkl"
c["predict"]["model"] = f"{d}/model_ft.pkl"
c["enr_avg_per_element"] = f"{d}/enr_avg_per_element.txt"
exec(extra)
json.dump(c, open(dst, "w"), indent=1)
PY
  sed "s/__NAME__/opt_$1/" > $T/run.sh <<'EOF'
#!/bin/bash
#SBATCH -J __NAME__
#SBATCH -p mig-3g.40gb
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -o slurm-%j.out
L=/proj/home/ibs/tdqh/hgpark/calc/battery/Na_cathode/Ni_doped_NMO/lsw
cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
T0=$(date +%s)
# 출력 각 줄 앞에 시각(HH:MM:SS) 추가 -> 진행률별 속도 계산용
$L/env/bin/python -u $L/calc_opt/tools/mem_wrap.py \
    $L/BAM-torch_DeNS/examples/bam-mp-dens/main_gpu.py input.json 2>&1 \
  | while IFS= read -r line; do printf '%s %s\n' "$(date +%T)" "$line"; done
echo "WALL_SEC $(( $(date +%s) - T0 ))"
EOF
  chmod +x $T/run.sh
  echo "--- $1: diff vs 원본"
  diff $L/calc/input.json $T/input.json | grep -E '^[<>]' || true
  (cd $T && sbatch run.sh)
}

make_test 01_baseline_re 'pass'
make_test 02_no_gradckpt 'c["grad_checkpoint"] = False'
squeue -u hgpark
