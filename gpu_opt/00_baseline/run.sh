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
