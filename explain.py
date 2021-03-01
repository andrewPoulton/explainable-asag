import fire
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import transformers
import captum.attr as attributions
# from captum.attr import (
#     IntegratedGradients,
#     InputXGradient)
from captum.attr import visualization
from configuration import load_configs_from_file
import dataset
import json
import os
from tqdm import tqdm

__CUDA__ = torch.cuda.is_available()

def load_model_from_disk(path):
    weights, config = torch.load(path, map_location='cpu')
    config = config['_items']
    mdl = transformers.AutoModelForSequenceClassification.from_pretrained(config['model_name'])
    mdl.load_state_dict(weights)
    print(f"Loaded model {config['model_name']} from {path} successfully.")
    return mdl, config

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

# def rank_tokens_by_attribution(batch, attributes, norm = 2, **kwargs):
#     norm = kwargs.get("norm", norm)
#     rank_order = attributes.norm(norm, dim = -1).argsort().squeeze()
#     ordered_tokens = batch.input.cpu().squeeze()[rank_order].numpy()[::-1].tolist()
#     return rank_order, ordered_tokens

def summarize(attr, aggr = 'norm', norm = 2):
    if aggr == 'norm':
        attr =  attr.norm(norm, dim = -1).squeeze(0)
    else:
        attr = attr.sum(dim=-1).squeeze(0)
    return attr.cpu().detach().numpy()


def explain_batch(attribution_method, model, batch, target = False, **kwargs):
    embeds = get_embeds(model, batch.input)
    with torch.no_grad():
        logits = model(inputs_embeds = embeds).logits.cpu().squeeze()
        pred = logits.argmax().item()
        pred_prob = torch.nn.functional.softmax(logits, dim=0).max().item()
    exp =  explainer(model, attribution_method)
    if kwargs.get("baselines", False):
        baseline = get_baseline(model, batch)
        kwargs["baselines"] = baseline
    if not target:
        target = pred
    if attribution_method == 'Occlusion':
        sliding_window_shape = (1,embeds.shape[-1])
        attr = exp.attribute(embeds, sliding_window_shape, target = target, additional_forward_args = model,  **kwargs)
    else:
        attr = exp.attribute(embeds, target = target, additional_forward_args = model,  **kwargs)
    return attr, pred, pred_prob


def explain(data_file, model_dir,  attribution_method):
    for file in os.scandir(model_dir):
        if file.name.endswith('.pt'):
            model_path = os.path.join(model_dir, file.name)
    model, config  = load_model_from_disk(model_path)
    model.eval()
    if __CUDA__:
        model.cuda()
    kwargs = load_configs_from_file('configs/explain.yml')["EXPLAIN"].get(attribution_method, {}) or {}
    if __CUDA__:
        num_workers = 8
    else:
        num_workers = 0
    expl_dataloader = dataset.dataloader(
        val_mode = True,
        data_file = data_file,
        data_source = config['data_source'],
        vocab_file = config['model_name'],
        num_labels = config['num_labels'],
        train_percent = 100,
        batch_size = 1,
        drop_last = False,
        num_workers = num_workers)
    tokenizer = expl_dataloader.dataset.tokenizer
    attr_L1_list = []
    attr_L2_list = []
    attr_sum_list = []
    tokens_list = []
    label_list = []
    prob_list = []
    pred_list = []
    attr_class_list = []
    attr_score_list = []
    # NOTE: This only works for batch_size = 1 and relies on it for now
    with tqdm(total=len(expl_dataloader.batch_sampler)) as pbar:
        pbar.set_description(f'Compute {attribution_method}:')
        for batch in expl_dataloader:
            label = batch.labels.item()
            if __CUDA__:
                batch.cuda()
            for target in range(config['num_labels']):
                attr, pred, pred_prob  = explain_batch(attribution_method, model, batch, target = target, **kwargs)
                attr_L1 = summarize(attr, aggr = 'norm', norm=1)
                attr_L2 = summarize(attr, aggr = 'norm', norm=2)
                attr_sum = summarize(attr, aggr = 'sum')
                attr_score = attr.sum().cpu().item()
                tokens = tokenizer.decode(batch.input.squeeze())
                attr_L1_list.append(attr_L1)
                attr_L2_list.append(attr_L2)
                attr_sum_list.append(attr_sum)
                tokens_list.append(tokens)
                label_list.append(label)
                prob_list.append(pred_prob)
                pred_list.append(pred)
                attr_class_list.append(target)
                attr_score_list.append(attr_score)
            batch.cpu()
            pbar.update(1)

    expl = pd.DataFrame({'label': label_list, 'pred': pred_list, 'pred_prob': prob_list,
                                   'attr_class': attr_class_list, 'attr_L1': attr_L1_list,'attr_L2': attr_L2_list, 'attr_sum': attr_sum_list,  'tokens': tokens_list})
    expl.to_pickle(os.path.join('explained',  config['name'] + '_' + config['data_source'] + '_' + model_dir  + '_' + attribution_method + '.pkl'))

    model.cpu()
    #return expl


if __name__=='__main__':
    fire.Fire(explain)
