#! /bin/bash
declare -a Models=(
#"bert-base"
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


for model in "${Models[@]}"
    do
        python train.py "$model" beetle --group 3
        wandb sync --sync-all
        wandb sync --clean
        rm -r wandb
done
