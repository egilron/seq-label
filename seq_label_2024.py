import os, sys, json
from dataclasses import dataclass, field
from typing import Optional
from datasets import  load_dataset, load_from_disk 
from pathlib import Path
import evaluate
import transformers
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Moved this inside again to have only one file
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("Numpy:", np.version.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)

transformers.logging.set_verbosity_warning()
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
Path("logs","jsons").mkdir( exist_ok=True, parents=True)
Path("logs","training_args").mkdir( exist_ok=True)
Path("logs","predictions").mkdir( exist_ok=True)
completed = os.listdir("logs/jsons")

# Parse from json file submitted as argument to the .py file
# We should pass only one argument to the script and it's the path to a json file,
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    print("\n\n\n***Loading config file:", sys.argv[1])
    config_path = Path(sys.argv[1]).resolve()
    config_name = config_path.name # including .json
    config_json = json.loads(config_path.read_text())
    args_dict = config_json["args_dict"]
    
else:
    print(sys.argv[0], "Made for getting one parameter, the json config file")
    if len(sys.argv) == 2:
        print("Got", sys.argv[1])
    sys.exit()

model_args, data_args, training_args = parser.parse_dict(args_dict)

Path("logs/training_args", config_name).write_text(json.dumps(training_args.to_dict()))
# Check for completed already
if config_name in completed:
    print(config_name, "seems to be completed. Exiting")
    sys.exit()
else: # Write a dummy in case we have multiple processes running
    Path("logs/jsons", config_name).write_text("LOCKED")


text_column_name = data_args.text_column_name
label_column_name = data_args.label_column_name
assert data_args.label_all_tokens == False, "Our script only labels first subword token"
# Load data from HF or disk. Don't trust the first character to be "/" for local.
use_local_data = config_json.get("local_dataset", False)
if use_local_data:
    dsd = load_from_disk(data_args.dataset_name)
else: 
    ds_version = "default"
    if "," in data_args.dataset_name: # Version / variant identifier
        ds_path, ds_version = data_args.dataset_name.split(",")
    else:
        ds_path = data_args.dataset_name
    dsd = load_dataset(ds_path.strip(),ds_version.strip() )

# Define what remote code we will trust
trust_remote = "norbert3" in model_args.model_name_or_path


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    sorted_labels = sorted(label_list,key=lambda name: (name[1:], name[0])) # Gather B and I
    return sorted_labels
label_list = get_label_list(dsd["train"][data_args.label_column_name]) # "tsa_tags"
label_to_id = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)
labels_are_int = False
label_list

# %%
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
    trust_remote_code=True
)
tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
if config.model_type in {"gpt2", "roberta"}:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

# %%
# Instanciate the model

model = AutoModelForTokenClassification.from_pretrained(
model_args.model_name_or_path,
from_tf=bool(".ckpt" in model_args.model_name_or_path),
config=config,
cache_dir=model_args.cache_dir,
revision=model_args.model_revision,
ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
trust_remote_code=trust_remote,
)


# %%

# print("Model's label2id:", model.config.label2id)
print(args_dict["task_name"], "Our label2id:", label_to_id)
# print("Our label list:  ", label_list)
# print("PretrainedConfig", PretrainedConfig(num_labels=num_labels).label2id)
assert (model.config.label2id == PretrainedConfig(num_labels=num_labels).label2id) or (model.config.label2id == label_to_id), "Model seems to have been fine-tuned on other labels already. Our script does not adapt to that."

# Set the correspondences label/ID inside the model config
model.config.label2id = {l: i for i, l in enumerate(label_list)}
model.config.id2label = {i: l for i, l in enumerate(label_list)}


# Preprocessing the dataset
# Padding strategy
padding = "max_length" if data_args.pad_to_max_length else False

# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        max_length=data_args.max_seq_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None or word_idx == previous_word_idx :
                label_ids.append(-100)
            # We set the label for the first token of each word only.
            else : #New word
                label_ids.append(label_to_id[label[word_idx]])
            # We do not keep the option to label the subsequent subword tokens here.

            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


with training_args.main_process_first(desc="train dataset map pre-processing"):
    train_dataset = dsd["train"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file= False,
        desc="Running tokenizer on train dataset",
    )
with training_args.main_process_first(desc="validation dataset map pre-processing"):
    eval_dataset = dsd["validation"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )
with training_args.main_process_first(desc="validation dataset map pre-processing"):
    predict_dataset = dsd["test"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on test dataset",
    )



data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

# Metrics
metric = evaluate.load("seqeval") # 

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if data_args.return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

# %%
print(args_dict["task_name"],"Ready to train. Train dataset labels are now:", train_dataset.column_names)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


if training_args.do_train:
    train_result = trainer.train(resume_from_checkpoint=False)
    metrics = train_result.metrics
    trainer.save_model(os.path.join(args_dict["output_dir"], "best_model"))  # Saves the tokenizer too for easy upload
    metrics["train_samples"] =  len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
else:
    metrics = {"Eval_only":True}

epoch_metrics = trainer.state.log_history
print(epoch_metrics)
Path("logs","jsons", config_name ).write_text( json.dumps(epoch_metrics))
# Evaluate
print("\nEvaluation,",model_args.model_name_or_path)
 
# Predict
trainer_predict = trainer.predict(predict_dataset, metric_key_prefix="predict")
predictions, labels, metrics = trainer_predict
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
 
gold = predict_dataset[args_dict["label_column_name"]] # Note: Will not work if dataset has ints, and the text labels in metadata
for g, pred in zip(gold,true_predictions ):
    assert len(g) == len(pred), (len(g) , len(pred))

try:
    if data_args.return_entity_level_metrics:
        seqeval_f1 =  metrics["predict_overall_f1"]
    else:
        seqeval_f1 =  metrics["predict_f1"]
except:
    seqeval_f1 = 0


trainer.log_metrics("predict", metrics)
trainer.save_metrics("predict", metrics)

# Adding processing the results here
epoch_eval = [ee for ee in epoch_metrics if "eval_f1" in ee]
epoch_eval = sorted(epoch_eval, key = lambda l: l["eval_f1"], reverse=True)
best_epoch = int(epoch_eval[0]["epoch"])

config_json["best_epoch"] = best_epoch
config_json["train_epochs_val"] = epoch_eval

config_path.write_text(json.dumps(config_json))
Path("logs/predictions", config_name).write_text(json.dumps(true_predictions))
print(f"Train and save best epoch to {sys.argv[1]} completed. F1:", seqeval_f1)