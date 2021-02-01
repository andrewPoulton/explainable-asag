
CONFIGS = {
    # size variation
    "bert-base-uncased":[],
    "bert-large-uncased":[],
    "roberta-base-uncased": [],
    "roberta-large-uncased": [],
    "albert-base-v2": [],
    "albert-large-v2": [],

    #distilled versions
    "distilbert-base-uncased":[],
    "distilroberta-base":[],
    
    #question-answering pretraining
    "twmkn9/distilbert-base-uncased-squad2": [],
    "roberta-base-squad2": [],
    "twmkn9/distilroberta-base-squad2":[],
    "twmkn9/bert-base-uncased-squad2": [],
    "twmkn9/albert-base-v2-squad2": [],

    #sentence similarity pretraining
    "sentence-transformers/ce-roberta-large-stsb": [],
    "sentence-transformers/ce-distilroberta-base-stsb": []

}

TRAIN_BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
LEARN_RATE = 1e-5
EPOCHS = 24
WARMUP_STEPS = 1024
SEQUENCE_LENGTH = 512