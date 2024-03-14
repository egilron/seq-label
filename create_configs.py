import os, sys, json
from pathlib import Path
from datetime import datetime
import itertools
timestamp = datetime.now().strftime("%m%d%H%M")

# Create the json files with model args, data args and training args for the experiments.
# First, set a default
# Then provide the lists of options
# Then provide any manual additional settings

CONFIG_ROOT = "configs"

# Use same keys here as in ds:

label_col = {"tsa-bin": "tsa_tags",
    "tsa-intensity": "tsa_tags", 
    "elsa-intensity": "elsa_labels",
    "elsa-polarity": "elsa_labels",
    "ner2":"ner_labels" }

# Settings dependent on where we are working
# These must be set manually:
ms, ds, local_out_dir = None, None, None
# The models addresses may be local folders, and then they need to be location-specific
# If not, they can be specified here above


ms = { 
    "norbert3-large":"ltg/norbert3-large", 
    "nb-bert-large":"NbAiLab/nb-bert-large",
    "nb-bert_base": "NbAiLab/nb-bert-base",
      }




# ds = {"tsa-bin": "data/tsa_binary",
    #   "tsa-intensity":"data/tsa_intensity" }
ds = { # "tsa-bin": "ltg/norec_tsa,default",
     # "tsa-intensity":"ltg/norec_tsa,intensity",
     "elsa-polarity": "data/elsa-dataset_seqlabel",
    #  "ner2" : "data/ner_2cat"
      }
LOCAL_DATASET = list(ds.values())[0].startswith("data/") # We want to ba able to force this here, and not depend on the character pattern if needed.


local_out_dir = None
WHERE = "saga"
if len(sys.argv) == 2:
    WHERE = sys.argv[1]
print("WHERE:", WHERE)



if WHERE == "hp":
    local_out_dir = "~/tsa_testing"

if WHERE == "saga":
    local_out_dir = "/cluster/work/users/egilron/finetunes/"

if WHERE == "fox":
    local_out_dir = "/cluster/work/projects/ec30/egilron/tsa-hf"


if WHERE == "lumi":
    local_out_dir = "/scratch/project_465000144/egilron/sq_label"

assert not any([e is None for e in [ms, ds, local_out_dir]]), "ms, ds, and local_out_dir need values set above here"



# Add training args as needed
default = {
    "model_name_or_path": None, #ms["brent0"] ,
    "dataset_name": None,
    "seed": 101,
    "per_device_train_batch_size": 32,
    "task_name": "ner", # Change this in iteration if needed
    "output_dir": local_out_dir,   # Add to this in iteration to avoid overwriting
    "overwrite_cache": True,
    "overwrite_output_dir": True,
    "do_train": True,
    "num_train_epochs": 12,
    # "num_warmup_steps": 50, # Must go to the optimizer
    "do_eval": True,
    "return_entity_level_metrics": False, # True,
    "use_auth_token": False,
    "logging_strategy": "epoch",
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch", # 
    "save_total_limit": 1,
    "load_best_model_at_end": True, #  
    "label_column_name": None,
    "disable_tqdm": True,
    "report_to": None,
    "do_predict": True,
    "text_column_name": "tokens"
}



# Iterations: design this according to needs
seeds = [101, 202, 303]
# seed = seeds[0]
# for task in ds.keys(): #["tsa-bin"]: #
task = list(ds)[0]
for seed in seeds:
    experiments = [] # List of dicts, one dict per experiments: Saves one separate json file each
    for i, ( b_size, l_rate) in enumerate(itertools.product( [32,64], [  1e-5, 5e-5])):
        for m_name, m_path in ms.items():
            exp = default.copy()
            exp ["per_device_train_batch_size"] = b_size
            exp["learning_rate"] = l_rate
            exp["seed"] = seed
            exp["trust_remote_code"] = m_path.startswith("ltg/")
            exp["model_name_or_path"] = m_path
            exp["dataset_name"] = ds [task]
            exp["task_name"] = f"{timestamp}_{task}_{m_name}" # Add seed in name if needed
            exp["output_dir"] = os.path.join(default["output_dir"], exp["task_name"] )
            exp["label_column_name"] = label_col.get(task, "")

            experiments.append({"timestamp":timestamp, "num_seeds": len(seeds),
                                "task":task, "model_shortname": m_name,
                                "machinery":WHERE,
                                "local_dataset": LOCAL_DATASET,
                                "args_dict":exp, "best_epoch":None})

    for i, exp in enumerate(experiments): # Move this with the experiments list definition to make subfolders
        args_dict = exp["args_dict"] # The args_dict was initially the entire exp

        save_path = Path(CONFIG_ROOT,WHERE, args_dict["task_name"]+"_"+str(i).zfill(2)+"_"+str(seed)+".json")
        save_path.parent.mkdir( parents=True, exist_ok=True)
        print(str(save_path))
        with open(save_path, "w") as wf:
            json.dump(exp, wf)


