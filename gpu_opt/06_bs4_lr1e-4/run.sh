#!/bin/bash
#SBATCH -J opt_06_bs4_lr1e-4
#SBATCH -p mig-3g.40gb
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:50:00
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
