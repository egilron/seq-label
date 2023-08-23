#!/bin/bash
#SBATCH --job-name=ner2
#SBATCH --mail-type=FAIL
#SBATCH --account=ec30
#SBATCH --partition=accel_long    # accel_long should give A100's
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00     
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --output=slurmfiles/%j.out

set -o errexit  # Recommended for easier debugging
source /etc/profile
module purge
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
python -m venv ~/venvs/transformers  # --clear # Comment out the --clear when you already have a clean install
source ~/venvs/transformers/bin/activate
# pip install -r requirements_addon.txt --quiet  # Comment out when you already have a clean install
# pip install accelerate -U
export TRANSFORMERS_CACHE=/cluster/work/projects/ec30/IN5550/.cache/

# d=$1
DIR=configs/fox/*
# filter="08231013"
for f in $DIR
do
    echo $f
    python seq_label.py $f 
done

