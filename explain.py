## Implement:
## TODO: inputxgrad
## TODO: inputXgradient
## WAIT: occlusion
## TODO: saliency
## TODO: integrated gradients


import fire
import numpy as np

import torch
import torch.nn as nn

import transformers
import captum.attr as attributions
# from captum.attr import (
#     IntegratedGradients,
#     InputXGradient)

import dataset

def construct_ref_ids(input_ids, cls_token_id = 101, sep_token_id = 102, ref_token_id = 0):
    ref_ids = torch.zeros_like(input_ids)
    if ref_token_id !=0:
        ref_ids.fill_(ref_token_id)
    ref_ids[input_ids==cls_token_id] = cls_token_id
    ref_ids[input_ids==sep_token_id] = sep_token_id
    return ref_ids

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

def rank_tokens_by_attribution(batch, attributes, norm = 2):
    rank_order = attributes.norm(norm, dim = -1).argsort().squeeze()
    ordered_tokens = batch.input.squeeze()[rank_order]
    return rank_order, ordered_tokens

def explain_batch(attibution_method, model, batch, **kwargs):
    kwargs = kwargs.get(attibution_method, {})
    embeds = get_embeds(model, batch.input)
    with torch.no_grad():
        pred = model(inputs_embeds = embeds).logits.squeeze().argmax().item()

    exp =  explainer(model, attibution_method)
    if kwargs.get("baselines", False):
        baseline = get_baseline(model, batch)
        kwargs["baselines"] = baseline
    attr = exp.attribute(embeds, target = pred, additional_forward_args = model, **kwargs)
    return attr

def explain_validation_data():
    val_dataloader = dataset.dataloader(val_mode = True, batch_size = 1, num_workers = 1)

    for b in val_dataloader:
        break
    model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)
    model.eval()
    embeds = get_embeds(model, b.input)
    baseline = get_baseline(model, b)
    with torch.no_grad():
        pred = model(input_embeds = embeds).logits.squeeze().argmax().item()

    exp =  explainer(model)
    attr = explainer.attribute(embeds, baselines = baseline, target = pred, additional_forward_args = model)
    return attr


if __name__=='__main__':
    fire.Fire(explain_validation_data)
