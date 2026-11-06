#!/bin/bash
#SBATCH -J nnmo_dens_ft
#SBATCH -p mig-3g.40gb
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 3-00:00:00
#SBATCH -o slurm-%j.out
L=/proj/home/ibs/tdqh/hgpark/calc/battery/Na_cathode/Ni_doped_NMO/lsw
cd $L/calc
export OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv
$L/env/bin/python -u $L/BAM-torch_DeNS/examples/bam-mp-dens/main_gpu.py input.json
