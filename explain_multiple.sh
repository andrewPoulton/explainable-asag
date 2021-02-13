#! /bin/bash
declare -a ModelPaths=(
    "model/roberta-base-best_f1.pt"
)

declare -a AttributionMethods=(
    "IntegrtedGradients"
    "InputXGradient"
    "Saliency"
    "GradientShap"
)

for path in "${ModelPaths[@]}"
do
    for method in "${AttributionMethods[@]}"
    do
        python explain.py "data/flat_semeval5way_test.csv" "$path" "$method"
    done
done
