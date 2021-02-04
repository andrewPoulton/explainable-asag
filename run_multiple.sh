#! /bin/bash
python run.py --experiment 'bert-base'
wandb gc
python run.py --experiment 'bert-large'
wandb gc
python run.py --experiment 'roberta-base'
wandb gc
python run.py --experiment 'roberta-large'
wandb gc
python run.py --experiment 'albert-base'
wandb gc
python run.py --experiment 'albert-large'
wandb gc
python run.py --experiment 'distilbert-base-uncased'
wandb sync --clean
python run.py --experiment 'distilroberta'
wandb sync --clean
python run.py --experiment 'distilbert-base-squad2'
wandb sync --clean
python run.py --experiment 'roberta-base-squad2'
wandb sync --clean
python run.py --experiment 'distilroberta-base-squad2'
wandb sync --clean
python run.py --experiment 'bert-base-squad2'
wandb sync --clean
python run.py --experiment 'albert-base-squad2'
wandb sync --clean
