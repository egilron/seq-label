import os, json
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
    "tsa-intensity": "tsa_tags"}

# Settings dependent on where we are working
# These must be set manually:
ms, ds, local_out_dir = None, None, None
# The models addresses may be local folders, and then they need to be location-specific
# If not, they can be specified here above


ms = { "NorBERT_3_small": "ltg/norbert3-small",
    "NorBERT_3_base": "ltg/norbert3-base", 
    "XLM-R_base": "xlm-roberta-base",
    "NB-Roberta_base": "NbAiLab/nb-roberta-base-ncc-plus-scandi-1e4",
    "NB-BERT_large": "NbAiLab/nb-bert-large",
    "NorBERT_3_large": "ltg/norbert3-large",
      }

# Comma in path will be used to extract config alternative from HF
ds = {"tsa-bin": "data/tsa_binary",
      "tsa-intensity":"data/tsa_intensity" }

local_out_dir = None

WHERE = "lumi"
if WHERE == "hp":
    local_out_dir = "~/tsa_testing"


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
    "task_name": "tsa", # Change this in iteration if needed
    "output_dir": local_out_dir,   # Add to this in iteration to avoid overwriting
    "overwrite_cache": True,
    "overwrite_output_dir": True,
    "do_train": True,
    "num_train_epochs": 16,
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
for task in ["tsa-bin"]: #ds.keys():
    experiments = [] # List of dicts, one dict per experiments: Saves one separate json file each
    for i, ( b_size, l_rate) in enumerate(itertools.product( [ 64], [ 5e-5, 1e-5, 1e-6])):
        for m_name, m_path in ms.items():
            exp = default.copy()
            exp ["per_device_train_batch_size"] = b_size
            exp["learning_rate"] = l_rate

            exp["model_name_or_path"] = m_path
            exp["dataset_name"] = ds [task]
            exp["task_name"] = f"{timestamp}_{task}_{m_name}" # Add seed in name if needed
            exp["output_dir"] = os.path.join(default["output_dir"], exp["task_name"] )
            exp["label_column_name"] = label_col.get(task, "")

            experiments.append({"timestamp":timestamp, "num_seeds": 5,
                                "task":task, "model_shortname": m_name,
                                "machinery":WHERE,"args_dict":exp, "best_epoch":None})

    for i, exp in enumerate(experiments): # Move this with the experiments list definition to make subfolders
        args_dict = exp["args_dict"] # The args_dict was initially the entire exp

        save_path = Path(CONFIG_ROOT,WHERE, args_dict["task_name"]+"_"+str(i).zfill(2)+".json")
        save_path.parent.mkdir( parents=True, exist_ok=True)
        print(str(save_path))
        with open(save_path, "w") as wf:
            json.dump(exp, wf)

    # Standalone testing
    # save_path = Path(CONFIG_ROOT,"standalone", args_dict["task_name"]+".json")
    # save_path.parent.mkdir( parents=True, exist_ok=True)
    # with open(save_path, "w") as wf:
    #     json.dump(args_dict, wf)


