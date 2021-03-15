#! /bin/bash
declare -a Runs=(
#1z15ocj6 	# albert-base-squad2 beetle-1
1r5y32j9 	# bert-base-squad2 beetle-1
2n3tv79j 	# distilroberta-base-squad2 beetle-1
12tat8nu 	# roberta-base-squad2 beetle-1
26345i1e 	# distilbert-base-squad2 beetle-1
2kg72v22 	# distilroberta beetle-1
hhvjxvr4 	# distilbert-base beetle-1
7rusno3e 	# albert-large beetle-1
#37zy0hai 	# albert-base beetle-1
2eyxkvx0 	# roberta-large beetle-1
#2owzw8vf 	# roberta-base beetle-1
#9c9fsnl5 	# bert-large beetle-1
1hghnl1f 	# bert-base beetle-1
18i0o3x5 	# albert-base-squad2 beetle-2
#307twu8l 	# bert-base-squad2 beetle-2
#1bqcxmlf 	# distilroberta-base-squad2 beetle-2
ay0ti5uf 	# roberta-base-squad2 beetle-2
#aipicd1s 	# distilbert-base-squad2 beetle-2
2clxn96i 	# distilroberta beetle-2
#2sbg632j 	# distilbert-base beetle-2
5bw20ac2 	# albert-large beetle-2
12xwwjt8 	# albert-base beetle-2
2ht4csf5 	# roberta-large beetle-2
2k9zrrkb 	# roberta-base beetle-2
38dwjr5w 	# bert-large beetle-2
hyd7o6ib 	# bert-base beetle-2
4b47e7sq 	# albert-base-squad2 beetle-3
21hrqqcx 	# bert-base-squad2 beetle-3
3b2oms9i 	# distilroberta-base-squad2 beetle-3
#rsuxma90 	# roberta-base-squad2 beetle-3
32p86rtw 	# distilbert-base-squad2 beetle-3
#2nxwh65v 	# distilroberta beetle-3
2zg2ng33 	# distilbert-base beetle-3
xlaiu1tp 	# albert-large beetle-3
ce8hfp9w 	# albert-base beetle-3
#xpdm71ti 	# roberta-large beetle-3
1xxv2t68 	# roberta-base beetle-3
2yrdjb9g 	# bert-large beetle-3
#1fkyddwv 	# bert-base beetle-3
)

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
