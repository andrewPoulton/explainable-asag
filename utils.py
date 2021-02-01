import json
import torch
from transformers import AutoModelForSequenceClassification
from types import SimpleNamespace

def load_config(cfg):
    config = json.load(open(cfg, 'r'))
    return SimpleNamespace(**config)


def configure_model(model, config):
    pass

def init_model(config):
    model = AutoModelForSequenceClassification.from_pretrained(config.model_path)
    emb = torch.nn.Embedding(config.type_vocab_size, model.config.hidden_size)
    emb.weight.data.normal_(mean=0.0, std = model.config.initializer_range)
    set_token_type_embeddings(emb, model)
     

    return model

def set_token_type_embeddings(embeddings, module):
    available_embeddings = [attr for attr in dir(module) if "word_embeddings" in attr]
    if available_embeddings:
        setattr(module,"token_type_embeddings", embeddings)
        return
    for _, child in module.named_children():
        set_token_type_embeddings(embeddings, child)
