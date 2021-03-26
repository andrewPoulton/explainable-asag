#! /bin/bash
python evaluateDC.py attributions/beetle-1
python evaluateDC.py attributions/scientsbank-1
#python evaluateDC.py attributions/beetle-token_types-1
#python evaluateDC.py attributions/scientsbank-token_types-1

python evaluateRC.py attributions/beetle-1 attributions/beetle-2
python evaluateRC.py attributions/scientsbank-1 attributions/scientsbank-2
python evaluateRC.py attributions/scientsbank-token_types-1 attributions/scientsbank-token_types-2
python evaluateRC.py attributions/beetle-token_types-1 attributions/beetle-token_types-2
