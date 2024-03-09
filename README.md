# seq-label
Basic sequence labelling NER-style based on Huggingface Transformers code examples. This task of assigning one label to each token (Not sub-word token, but to each word in the pre-tokenized text) is now called "token-classification". The script here is simplified, with several options removed, or hard-coded. The HuggingFace template has it all.  
We have added some tweaks needed to run NorBERT3 with a Trainer. That may not be needed anymore when you read this.  
We use this repo for ongoing work, and that may be visible in a not-so-tidy contents. We use these scripts on a number of compute resources, and the setup is adjusted to that situation.
## Usage
The core file is `seq_label.py` which takes one json config-file as argument. The config file(s) are created with `create_fine-tune_json.py`. This file needs to be read and understood before running it. It can create multiple files, and a bash script can be used to iterate these as argument for `seq_label.py`. e.g.
```
DIR=configs/hp/*
for f in $DIR
do
   echo $f
   python seq_label.py $f 
done
```
## ELS-modelling
FIrst success, interactive:
```
***Loading config file: configs/saga/03080957_elsa-intensity_NB-BERT_base_00.json
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
Some weights of BertForTokenClassification were not initialized from the model checkpoint at NbAiLab/nb-bert-base and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
03080957_elsa-intensity_NB-BERT_base Our label2id: {'O': 0, 'B-Negative_Slight': 1, 'I-Negative_Slight': 2, 'B-Negative_Standard': 3, 'I-Negative_Standard': 4, 'B-Neutral': 5, 'I-Neutral': 6, 'B-Positive_Slight': 7, 'I-Positive_Slight': 8, 'B-Positive_Standard': 9, 'I-Positive_Standard': 10}
Running tokenizer on train dataset:   0%|                                                                                        | 0/8570 [00:00<?, ? examples/s]Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Running tokenizer on train dataset: 100%|███████████████████████████████████████████████████████████████████████████| 8570/8570 [00:03<00:00, 2558.52 examples/s]
Running tokenizer on validation dataset: 100%|██████████████████████████████████████████████████████████████████████| 1513/1513 [00:00<00:00, 2226.94 examples/s]
Running tokenizer on test dataset: 100%|████████████████████████████████████████████████████████████████████████████| 1252/1252 [00:00<00:00, 1703.19 examples/s]
03080957_elsa-intensity_NB-BERT_base Ready to train. Train dataset labels are now: ['sent_id', 'tokens', 'elsa_labels', 'entity', 'input_ids', 'token_type_ids', 'attention_mask', 'labels']
{'loss': 0.08, 'grad_norm': 0.43012261390686035, 'learning_rate': 2.5e-05, 'epoch': 1.0}                                                                         
 50%|█████████████████████████████████████████████████████████████                                                             | 268/536 [01:10<01:00,  4.45it/s/cluster/work/users/egilron/venvs/transformers/lib/python3.10/site-packages/seqeval/metrics/v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
{'eval_loss': 0.04710279777646065, 'eval_precision': 0.6592356687898089, 'eval_recall': 0.6666666666666666, 'eval_f1': 0.6629303442754203, 'eval_accuracy': 0.9862039316840583, 'eval_runtime': 5.4015, 'eval_samples_per_second': 280.105, 'eval_steps_per_second': 35.175, 'epoch': 1.0}                                        
 81%|███████████████████████████████████████████████████████████████████████████████████████████████████▏                      | 436/536 [02:03<00:26,  3.83it/s]
 82%|███████████████████████████████████████████████████████████████████████████████████████████████████▍                      | 43 82%|███████████████████████████████████████████████████████████████████████████▏                | 438/536 [02:03<00:24,  3.97it/s]{'loss': 0.0351, 'grad_norm': 0.31482335925102234, 'learning_rate': 0.0, 'epoch': 2.0}                                             
100%|████████████████████████████████████████████████████████████████████████████████████████████| 536/536 [02:28<00:00,  4.55it/s/cluster/work/users/egilron/venvs/transformers/lib/python3.10/site-packages/seqeval/metrics/v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
{'eval_loss': 0.047243136912584305, 'eval_precision': 0.651702786377709, 'eval_recall': 0.677938808373591, 'eval_f1': 0.664561957379637, 'eval_accuracy': 0.9863211787235706, 'eval_runtime': 5.3206, 'eval_samples_per_second': 284.366, 'eval_steps_per_second': 35.71, 'epoch': 2.0}
{'train_runtime': 158.4339, 'train_samples_per_second': 108.184, 'train_steps_per_second': 3.383, 'train_loss': 0.05750848997884722, 'epoch': 2.0}                                                                                                                    
100%|████████████████████████████████████████████████████████████████████████████████████████████| 536/536 [02:38<00:00,  3.38it/s]
***** train metrics *****
  epoch                    =        2.0
  train_loss               =     0.0575
  train_runtime            = 0:02:38.43
  train_samples            =       8570
  train_samples_per_second =    108.184
  train_steps_per_second   =      3.383
[{'loss': 0.08, 'grad_norm': 0.43012261390686035, 'learning_rate': 2.5e-05, 'epoch': 1.0, 'step': 268}, {'eval_loss': 0.04710279777646065, 'eval_precision': 0.6592356687898089, 'eval_recall': 0.6666666666666666, 'eval_f1': 0.6629303442754203, 'eval_accuracy': 0.9862039316840583, 'eval_runtime': 5.4015, 'eval_samples_per_second': 280.105, 'eval_steps_per_second': 35.175, 'epoch': 1.0, 'step': 268}, {'loss': 0.0351, 'grad_norm': 0.31482335925102234, 'learning_rate': 0.0, 'epoch': 2.0, 'step': 536}, {'eval_loss': 0.047243136912584305, 'eval_precision': 0.651702786377709, 'eval_recall': 0.677938808373591, 'eval_f1': 0.664561957379637, 'eval_accuracy': 0.9863211787235706, 'eval_runtime': 5.3206, 'eval_samples_per_second': 284.366, 'eval_steps_per_second': 35.71, 'epoch': 2.0, 'step': 536}, {'train_runtime': 158.4339, 'train_samples_per_second': 108.184, 'train_steps_per_second': 3.383, 'total_flos': 655433565619260.0, 'train_loss': 0.05750848997884722, 'epoch': 2.0, 'step': 536}]

Evaluation, NbAiLab/nb-bert-base
 97%|█████████████████████████████████████████████████████████████████████████████████████████   | 152/157 [00:03<00:00, 53.16it/s]/cluster/work/users/egilron/venvs/transformers/lib/python3.10/site-packages/seqeval/metrics/v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
100%|████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:04<00:00, 35.57it/s]
***** predict metrics *****
  predict_accuracy           =     0.9874
  predict_f1                 =     0.6725
  predict_loss               =      0.043
  predict_precision          =     0.6718
  predict_recall             =     0.6732
  predict_runtime            = 0:00:04.43
  predict_samples_per_second =    282.167
  predict_steps_per_second   =     35.384
Train and save best epoch to configs/saga/03080957_elsa-intensity_NB-BERT_base_00.json completed. F1: 0.6725082146768894
```