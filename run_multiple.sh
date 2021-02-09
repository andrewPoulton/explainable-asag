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

for g in 1 2 3
do
    for model in $Models
    do
        python run.py "$model" "$@" --group $g
        wandb sync --sync-all
        wandb sync --clean
        rm -r wandb
    done
done
