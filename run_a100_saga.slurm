#!/bin/bash
#SBATCH --job-name=elsa-seq
#SBATCH --mail-type=FAIL
#SBATCH --account=nn9851k
#SBATCH --partition=a100  
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00     
#SBATCH --mem-per-cpu=24G
#SBATCH --nodes=1
#SBATCH --output=slurmfiles/%j.out

set -o errexit  # Recommended for easier debugging
source /etc/profile


export HF_HOME=/cluster/work/users/egilron/.cache
module --quiet purge
module --force swap StdEnv Zen2Env
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8 
# module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8
# module load nlpl-huggingface-hub/0.19.4-foss-2022b-Python-3.10.8
# module load nlpl-datasets/2.15.0-foss-2022b-Python-3.10.8
# module load nlpl-accelerate/0.24.1-foss-2022b-Python-3.10.8
# deactivate
# python -m venv /cluster/work/users/egilron/venvs/transformers_a100 --clear
source /cluster/work/users/egilron/venvs/transformers_a100/bin/activate
# pip install -r requirements_addon.txt

FOLDER="/cluster/work/users/egilron/seq-label_github/configs/saga"

# filter="03081014_elsa-intensity_NorBERT_3_base*.json"
# filter="03081014_elsa-intensity_XLM-R_base*.json"
# filter="03081014_elsa-intensity_NB-BERT_base*.json"
# filter="03081014_elsa-intensity_NB-BERT_large*.json"
filter="03081014_elsa-intensity_NorBERT_3_large*.json"



jsons=$(find $FOLDER -name $filter); #  -name '${filter}*'

for f in $jsons
do
    echo $f
    echo $f > bash-out.txt
    python seq_label.py $f
done