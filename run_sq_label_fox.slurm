#!/bin/bash
#SBATCH --job-name=tsa_norbert
#SBATCH --mail-type=FAIL
#SBATCH --account=ec30
#SBATCH --partition=accel_long    # accel_long should give A100's
#SBATCH --gres=gpu:1
#SBATCH --time=9:00:00     
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --output=slurmfiles/%j.out

set -o errexit  # Recommended for easier debugging
source /etc/profile
module purge
# module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
python -m venv ~/venvs/transformers # --clear # Comment out the --clear when you already have a clean install
source ~/venvs/transformers/bin/activate
# pip install -r requirements_addon.txt --quiet  # Comment out when you already have a clean install
# pip install accelerate -U
# export TRANSFORMERS_CACHE=/cluster/work/projects/ec30/egilron/.cache/
export HF_HOME=/cluster/work/projects/ec30/egilron/.cache/
# d=$1


FOLDER="/fp/homes01/u01/ec-egilron/sqlabel-github/configs/fox/"

filter="01170944_tsa-intensity*.json"


jsons=$(find $FOLDER -name $filter); #  -name '${filter}*'

for f in $jsons
do
    echo $f
    echo $f > bash-out.txt
    python seq_label.py $f
done



