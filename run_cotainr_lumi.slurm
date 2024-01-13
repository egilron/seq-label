#!/usr/bin/env bash
#
# A LUMI SLURM batch script for the LUMI PyTorch single GPU test example from
# https://github.com/DeiC-HPC/cotainr
#
#SBATCH --job-name=singlegpu_gputest_example
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=small-g
#SBATCH --time=01:05:00
#SBATCH --account=project_465000144

srun singularity exec sqlabel24.sif python3 seq_label.py configs/lumi/01121628_tsa-bin_XLM-R_base_02.json