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
