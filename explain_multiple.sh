#! /bin/bash
declare -a Runs=(
    1fkyddwv #bert-base-beetle-3
    9c9fsnl5 #bert-large-beetle-1
    2owzw8vf #roberta-base-beetle-1
    xpdm71ti #roberta-large-beetle-3
    37zy0hai #albert-base-beetle-1
    2sbg632j #distilbert-beetle-2
    2nxwh65v #distilroberta-beetle-3
    307twu8l #bert-base-squad-beetle-2
    rsuxma90 #roberta-squad-beetle-3
    1z15ocj6 #albert-suqad-beetle-1
    aipicd1s #distilbert-squad-beetle-2
    1bqcxmlf #distilroberta-squad-beetle-2
)

declare -a AttributionMethods=(
    "IntegratedGradients"
    "InputXGradient"
    "Saliency"
    "GradientShap"
    "GuidedBackprop"
)

for RUNID in "${Runs[@]}"
do
    mkdir $RUNID
    cd $RUNID
    wandb pull $RUNID -p explainable-asag -e sebaseliens
    cd ..
    for METHOD in "${AttributionMethods[@]}"
    do
        python explain.py "data/flat_semeval5way_test.csv" $RUNID $METHOD
    done
    rm -r $RUNID
done
