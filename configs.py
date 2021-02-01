from warnings import warn
from types import SimpleNamespace

# global config variables
TRAIN_BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
LEARN_RATE = 1e-5
EPOCHS = 24
WARMUP_STEPS = 1024
SEQUENCE_LENGTH = 512



# default config (overwritten by CONFIGS)
DEFAULT_CONFIG = {
    "model_path" : "bert-base-uncased",
    "train_batch_size" : TRAIN_BATCH_SIZE,
    "accumulation_steps" : ACCUMULATION_STEPS,
    "learn_rate" : LEARN_RATE,
    "epochs" : EPOCHS,
    "warmup_steps" : WARMUP_STEPS,
    "sequence_length" : SEQUENCE_LENGTH
}


# configs for experiments
CONFIGS = {
## size variation
    "bert-base": {
        "model_path" : "bert-base-uncased",
        },
    "bert-large" : {
        "model_path" : "bert-large-uncased",
    },
    "roberta-base" : {
        "model_path" : "roberta-base-uncased",
        },
    "roberta-large" : {
        "model_path" : "roberta-large-uncased",
        },
    "albert-base" : {
        "model_path" : "albert-base-v2",
        },
    "albert-large" : {
        "model_path" : "albert-large-v2",
        },
## distilled versions
    "distilbert-base-uncased" : {
        "model_path" : "distilbert-base-uncased",
        },
    "distilroberta" : {
        "model_path" : "distilroberta-base",
        },
## question-answering pretraining
    "distilbert-base-squad2" : {
        "model_path" : "twmkn9/distilbert-base-uncased-squad2",
        },
    "roberta-base" : {
        "model_path" : "roberta-base-squad2",
        },
    "distilroberta-base-squad2" : {
        "model_path" : "twmkn9/distilroberta-base-squad2",
        },
    "bert-base-squad2" : {
        "model_path" : "twmkn9/bert-base-uncased-squad2",
        },
    "albert-base-squad2" : {
        "model_path" : "twmkn9/albert-base-v2-squad2",
    },
## sentence similarity pretraining
    "roberta-large-stsb" : {
        "model_path" : "sentence-transformers/ce-roberta-large-stsb",
    },
    "distilroberta-base-stsb" : {
        "model_path" : "sentence-transformers/ce-distilroberta-base-stsb"
    }
}

def list_experiments():
    return list(CONFIGS.keys())

def load(experiment):
    config = DEFAULT_CONFIG
    try:
        config.update(CONFIGS[experiment])
        print(f"Loaded configs for experiment '{experiment}'")
    except  KeyError:
        warn("Invalid experiment!")
        print(f"The experiment '{experiment} is invalid. Experiments should be chosen from:")
        for i, valid in enumerate(list_experiments()):
            print(f"{i}.  {valid}")
        print("NB: Will use default settings instead!")
    return SimpleNamespace(**config)



