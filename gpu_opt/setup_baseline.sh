set -e
L=/proj/home/ibs/tdqh/hgpark/calc/battery/Na_cathode/Ni_doped_NMO/lsw
O=$L/calc_opt
T=$O/00_baseline
[ -e "$T" ] && { echo "EXISTS: $T  (abort)"; exit 1; }
mkdir -p $O/template $T

# 원본 보존용 템플릿 (읽기 전용 참고본)
cp -p $L/calc/input.json $L/calc/run.sh $L/calc/enr_avg_per_element.txt $O/template/
# 테스트 폴더: 모델 복사본 포함
cp -p $L/calc/model_ft.pkl $L/calc/enr_avg_per_element.txt $T/

$L/env/bin/python - "$L/calc/input.json" "$T/input.json" "$T" <<'PY'
import json, sys
src, dst, d = sys.argv[1:]
c = json.load(open(src))
c["NN"]["nepoch"] = 3
c["NN"]["fname_pkl"] = f"{d}/model_ft.pkl"
c["predict"]["model"] = f"{d}/model_ft.pkl"
c["enr_avg_per_element"] = f"{d}/enr_avg_per_element.txt"
json.dump(c, open(dst, "w"), indent=1)
PY

cat > $T/run.sh <<'EOF'
#!/bin/bash
#SBATCH -J opt_00_baseline
#SBATCH -p mig-3g.40gb
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o slurm-%j.out
L=/proj/home/ibs/tdqh/hgpark/calc/battery/Na_cathode/Ni_doped_NMO/lsw
cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
# GPU 메모리 기록 (15초 간격, MIG에서는 compute-apps 기준)
( while true; do echo "== $(date +%T)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; sleep 15; done ) > gpu_mem.log 2>&1 &
MON=$!
T0=$(date +%s)
$L/env/bin/python -u $L/BAM-torch_DeNS/examples/bam-mp-dens/main_gpu.py input.json
echo "WALL_SEC $(( $(date +%s) - T0 ))"
kill $MON
EOF
chmod +x $T/run.sh

echo "--- diff input.json (원본 vs baseline)"
diff $L/calc/input.json $T/input.json || true
ls -la $O $T
cd $T && sbatch run.sh
squeue -u hgpark
