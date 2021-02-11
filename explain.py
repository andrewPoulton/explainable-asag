import fire
import numpy as np

import torch
import torch.nn as nn

import transformers
import captum.attr as attributions
# from captum.attr import (
#     IntegratedGradients,
#     InputXGradient)
from configuration import load_configs_from_file
import dataset

__CUDA__ = torch.cuda.is_available()

def load_model_from_disk(path):
    weights, config = torch.load(path, map_location='cpu')
    mdl = transformers.AutoModelForSequenceClassification.from_pretrained(config['model_path'])
    mdl.load_state_dict(weights)
    return mdl

def get_word_embeddings(module):
    for attr_str in dir(module):
        if attr_str == "word_embeddings":
            return  getattr(module, attr_str)

    for n, ch in module.named_children():
        embeds = get_word_embeddings(ch)
        if embeds:
            return embeds

def get_embeds(model, inputs):
    return get_word_embeddings(model)(inputs)

def get_baseline(model, batch):
    baseline_inputs = torch.where(batch.token_type_ids.eq(3), torch.zeros_like(batch.input), batch.input)
    return get_embeds(model, baseline_inputs)




def explainer(model, attribution_method = "IntegratedGradients"):
    attribution_method = attributions.__dict__[attribution_method]
    def func(embeds, model):
        return model(inputs_embeds = embeds).logits
    return attribution_method(func)


def rank_tokens_by_attribution(batch, attributes, norm = 2, **kwargs):
    norm = kwargs.get("norm", norm)
    rank_order = attributes.norm(norm, dim = -1).argsort().squeeze()
    ordered_tokens = batch.input.cpu().squeeze()[rank_order].numpy().tolist()
    return rank_order, ordered_tokens

def explain_batch(attibution_method, model, batch, **kwargs):
    
    embeds = get_embeds(model, batch.input)
    with torch.no_grad():
        pred = model(inputs_embeds = embeds).logits.squeeze().argmax().item()

    exp =  explainer(model, attibution_method)
    if kwargs.get("baselines", False):
        baseline = get_baseline(model, batch)
        kwargs["baselines"] = baseline
    attr = exp.attribute(embeds, target = pred, additional_forward_args = model, **kwargs)
    return attr.detach()

def explain_validation_data(model_path, attribution_method, config):
    val_dataloader = dataset.dataloader(val_mode = True, batch_size = 1, num_workers = 1, data_source="beetle")
    model = load_model_from_disk(model_path)
    model.eval()
    ordered_tokens = []
    config = load_configs_from_file(config)["EXPLAIN"].get(attribution_method, {}) or {}
    for batch in val_dataloader:
        attr = explain_batch(attribution_method, model, batch, **config)
        _, ranked_tokens = rank_tokens_by_attribution(batch, attr, **config)
        # print(ranked_tokens)
    return ordered_tokens

if __name__=='__main__':
    fire.Fire(explain_validation_data)
