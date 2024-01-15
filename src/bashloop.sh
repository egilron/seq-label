#!/bin/bash
# https://stackoverflow.com/questions/2087001/how-can-i-process-the-results-of-find-in-a-bash-script
FOLDER="/fp/homes01/u01/ec-egilron/sqlabel-github/configs/fox/"

filter="01121625_tsa-bin_NB-Roberta_base*.json"
IFS=$'\n'
jsons=$(find $FOLDER -name $filter); #  -name '${filter}*'

for f in $jsons
do
    echo $f
done
unset IFS