export HF_HOME=/cluster/work/users/egilron/.cache
module --quiet purge
module --force swap StdEnv Zen2Env
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8 
# module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8
# module load nlpl-huggingface-hub/0.19.4-foss-2022b-Python-3.10.8
# module load nlpl-datasets/2.15.0-foss-2022b-Python-3.10.8
# module load nlpl-accelerate/0.24.1-foss-2022b-Python-3.10.8
deactivate
python -m venv /cluster/work/users/egilron/venvs/transformers_a100 --clear
source /cluster/work/users/egilron/venvs/transformers_a100/bin/activate
pip install -r requirements_addon.txt