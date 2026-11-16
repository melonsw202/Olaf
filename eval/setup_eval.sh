# 8번: test set 평가 (FT 모델 vs 사전학습 모델). 원본 모델은 읽기만 함.
set -e
L=/proj/home/ibs/tdqh/hgpark/calc/battery/Na_cathode/Ni_doped_NMO/lsw
E=$L/eval
PY=$L/env/bin/python
mkdir -p $E/tools
cat > $E/tools/run_eval.py <<'EOF'
"""공식 MultiheadEvaluator로 predict 실행 (학습 코드 수정 없음). 사용: python run_eval.py input.json"""
import json, sys, torch
from bam_torch.predicting.mh_evaluator import MultiheadEvaluator
json_data = json.load(open(sys.argv[1]))
torch.cuda.empty_cache()
MultiheadEvaluator(json_data).evaluate()
EOF
make_eval () {   # $1=폴더 $2=모델 경로
  T=$E/$1
  [ -e "$T" ] && { echo "EXISTS: $T (abort)"; exit 1; }
  mkdir -p $T
  cp -p $L/calc/enr_avg_per_element.txt $T/
  $PY - "$L/calc/input.json" "$T/input.json" "$T" "$2" <<'EOF'
import json, sys
src, dst, d, model = sys.argv[1:]
c = json.load(open(src))
c["NN"]["restart"] = False
c["NN"]["fname_pkl"] = f"{d}/unused_model.pkl"
c["predict"]["model"] = model
c["predict"]["fname_plog"] = "predict.out"
c["enr_avg_per_element"] = f"{d}/enr_avg_per_element.txt"
json.dump(c, open(dst, "w"), indent=1)
EOF
  sed "s/__NAME__/eval_$1/" > $T/run.sh <<'EOF'
#!/bin/bash
#SBATCH -J __NAME__
#SBATCH -p mig-3g.40gb
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -o slurm-%j.out
L=/proj/home/ibs/tdqh/hgpark/calc/battery/Na_cathode/Ni_doped_NMO/lsw
cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
T0=$(date +%s)
$L/env/bin/python -u $L/eval/tools/run_eval.py input.json
echo "WALL_SEC $(( $(date +%s) - T0 ))"
EOF
  echo "SUBMITTED $1 $(cd $T && sbatch --parsable run.sh)"
}
make_eval ft_epoch118 $L/calc/model_ft.pkl
make_eval pretrained $L/mlp_data_set/BAM-MP-DeNS/BAM-MP-DeNS-model.pkl
squeue -u hgpark -p mig-3g.40gb
