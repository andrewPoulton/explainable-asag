from warnings import warn
from types import SimpleNamespace

DEFAULT_CONFIG = {
    "model_path" : "bert-base-uncased",
    "learn_rate" : 1e-5,
    "max_epochs" : 24,
    "warmup_steps" : 1024,
#    "sequence_length" : 512,
    "num_labels" : 2,
    "batch_size" : 8,
    "total_steps": 8192,
    "num_labels" : 2
}

CONFIGS = {
## size variation
#1
    "bert-base": {
        "model_path" : "bert-base-uncased",
        },
#2
    "bert-large" : {
        "model_path" : "bert-large-uncased",
    },
#3
    "roberta-base" : {
        "model_path" : "roberta-base",
        },
#4
    "roberta-large" : {
        "model_path" : "roberta-large",
        },
#5
    "albert-base" : {
        "model_path" : "albert-base-v2",
        },
#6
    "albert-large" : {
        "model_path" : "albert-large-v2",
        },
## distilled versions
#7
    "distilbert-base-uncased" : {
        "model_path" : "distilbert-base-uncased",
        },
#8
    "distilroberta" : {
        "model_path" : "distilroberta-base",
        },
## question-answering pretraining
#9
    "distilbert-base-squad2" : {
        "model_path" : "twmkn9/distilbert-base-uncased-squad2",
        },
#10
    "roberta-base-squad2" : {
        "model_path" : "deepset/roberta-base-squad2",
        },
#11
    "distilroberta-base-squad2" : {
        "model_path" : "twmkn9/distilroberta-base-squad2",
        },
#12
    "bert-base-squad2" : {
        "model_path" : "twmkn9/bert-base-uncased-squad2",
        },
#13
    "albert-base-squad2" : {
        "model_path" : "twmkn9/albert-base-v2-squad2",
    },
## sentence similarity pretraining ## TODO: Fails! Deal with mismatch in classification head.
# #14
#     "roberta-large-stsb" : {
#         "model_path" : "sentence-transformers/ce-roberta-large-stsb",
#     },
# #15
#     "distilroberta-base-stsb" : {
#         "model_path" : "sentence-transformers/ce-distilroberta-base-stsb"
#     }
}

def list_experiments():
    return list(CONFIGS.keys())

def load(experiment):
    config = DEFAULT_CONFIG
    try:
        print(f"Load configs for experiment '{experiment}'.")
        config.update(CONFIGS[experiment])
    except  KeyError:
        warn("Invalid experiment!")
        print(f"The experiment '{experiment} is invalid. Experiments should be chosen from:")
        for i, valid in enumerate(list_experiments()):
            print(f"{i}.  {valid}")
        print("NB: Will use default settings instead!")
    return SimpleNamespace(**config)



