#!/usr/bin/env bash
#
# A LUMI SLURM batch script for the LUMI PyTorch single GPU test example from
# https://github.com/DeiC-HPC/cotainr
#
#SBATCH --job-name=cotain
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=small-g
#SBATCH --time=07:05:00
#SBATCH --account=project_465000144

export HF_HOME="/scratch/project_465000144/egilron/hfhome"




# One file
# srun singularity exec --bind /scratch/project_465000144/egilron:/scratch/project_465000144/egilron \
    # /scratch/project_465000144/egilron/sqlabel24.sif\
    # python3 seq_label.py configs/lumi/01121628_tsa-bin_XLM-R_base_02.json

DIR="configs/lumi/"
filter="01121628_tsa*"
# for f in "configs/fox/0825*";
for f in $DIR$filter;
do
    echo $f
    echo $f > bash-out.txt
    sleep 1 # Probably from debugging
    srun singularity exec --bind /scratch/project_465000144/egilron:/scratch/project_465000144/egilron \
    /scratch/project_465000144/egilron/sqlabel24.sif\
    python3 seq_label.py $f

done