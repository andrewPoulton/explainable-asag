#! /bin/bash
declare -a Runs=(
1egx860t 	# albert-base-squad2 scientsbank-1
30n99k6q 	# bert-base-squad2 scientsbank-1
3oysdwkz 	# distilroberta-base-squad2 scientsbank-1
2ba9sddh 	# roberta-base-squad2 scientsbank-1
1uyfnw01 	# distilbert-base-squad2 scientsbank-1
1rzchb68 	# distilroberta scientsbank-1
1lakool8 	# distilbert-base scientsbank-1
79fh7xu4 	# albert-large scientsbank-1
u4lvktg5 	# albert-base scientsbank-1
kuio0joh 	# roberta-large scientsbank-1
xtqwd2j0 	# roberta-base scientsbank-1
3m2qfbww 	# bert-large scientsbank-1
23tj1jnv 	# bert-base scientsbank-1
29xaqom2 	# albert-base-squad2 scientsbank-2
3ci6yiv7 	# bert-base-squad2 scientsbank-2
4qafstv7 	# distilroberta-base-squad2 scientsbank-2
1ya8ifek 	# roberta-base-squad2 scientsbank-2
21vr6xj1 	# distilbert-base-squad2 scientsbank-2
29lug2n0 	# distilroberta scientsbank-2
39q7aaxm 	# distilbert-base scientsbank-2
33siklf6 	# albert-large scientsbank-2
byly99en 	# albert-base scientsbank-2
3cwv9l2h 	# roberta-large scientsbank-2
1sjiw3ri 	# roberta-base scientsbank-2
2lqtrunf 	# bert-large scientsbank-2
3spj7e10 	# bert-base scientsbank-2
38ru3ayt 	# albert-base-squad2 scientsbank-3
27ot6q2h 	# bert-base-squad2 scientsbank-3
1gvvhr9e 	# distilroberta-base-squad2 scientsbank-3
37zdoxlf 	# roberta-base-squad2 scientsbank-3
37mxih4t 	# distilbert-base-squad2 scientsbank-3
28ytbh55 	# distilroberta scientsbank-3
vyg0gkqf 	# distilbert-base scientsbank-3
1ob2q968 	# albert-large scientsbank-3
1gg8ydfw 	# albert-base scientsbank-3
24ct9efg 	# roberta-large scientsbank-3
2tpaz9ju 	# roberta-base scientsbank-3
29u5lpqa 	# bert-large scientsbank-3
9h3wzj16 	# bert-base scientsbank-3)

declare -a AttributionMethods=(
    "InputXGradient"
    "Saliency"
    "IntegratedGradients"
    "GradientShap"
    "Occlusion"
)

for RUNID in "${Runs[@]}"
do
    mkdir -p $RUNID
    cd $RUNID
    wandb pull $RUNID -p explainable-asag -e sebaseliens
    cd ..
    for METHOD in "${AttributionMethods[@]}"
    do
        python explain.py "data/flat_semeval5way_test.csv" $RUNID $METHOD
    done
    rm -r $RUNID
done
