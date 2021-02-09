#! /bin/bash
declare -a Models=(
"bert-base"
"bert-large"
"roberta-base"
"roberta-large"
"albert-base"
"albert-large"
"distilbert-base"
"distilroberta"
"distilbert-base-squad2"
"roberta-base-squad2"
"distilroberta-base-squad2"
"bert-base-squad2"
"albert-base-squad2")
for model in $Models
do
    for g in 1 2 3
    do
        python run.py $model --group $g
        wandb sync --sync-all --clean
        python run.py $model from_scratch --group $g
        wandb sync --sync-all --clean
    done
done
