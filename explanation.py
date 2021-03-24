import torch
import transformers
import captum.attr as attributions
from wandbinteraction import load_model_from_run_id
from modelhandling import get_word_embeddings
from training import compute_logits
from configuration import load_configs_from_file
from tqdm import tqdm
import os
import gc
import pandas as pd
import pickle
from collections import defaultdict
import json

def get_embeds(model, inputs):
    return get_word_embeddings(model)(inputs)

def get_baseline(model, batch):
    baseline_inputs = torch.where(batch.token_type_ids.eq(3), torch.zeros_like(batch.input), batch.input)
    return get_embeds(model, baseline_inputs)

def explainer(model, attribution_method, token_types):
    attribution_method = attributions.__dict__[attribution_method]
    if token_types:
        def func(embeds, model, token_type_ids):
            return model(inputs_embeds = embeds, token_type_ids = token_type_ids).logits
    else:
        def func(embeds, model):
            return model(inputs_embeds = embeds).logits
    return attribution_method(func)

def summarize(attr, aggr):
    if aggr == 'L2':
        attr =  attr.norm(2, dim = -1).squeeze(0)
    elif aggr == 'L1':
        attr =  attr.norm(1, dim = -1).squeeze(0)
    elif aggr == 'sum':
        attr = attr.sum(dim=-1).squeeze(0)
    else:
        raise Exception('No valid aggregation method in "summarize" attributions.')
    return attr.cpu().detach().numpy().tolist()


def explain_batch(attribution_method, model, token_types, batch, target, **kwargs):
    embeds = get_embeds(model, batch.input)
    exp =  explainer(model, attribution_method, token_types)

    if attribution_method == 'Occlusion':
        sliding_window_shape = (1,embeds.shape[-1])
        if token_types:
            attr = exp.attribute(embeds, sliding_window_shape, target = target, additional_forward_args = (model, batch.token_type_ids),  **kwargs)
        else:
            attr = exp.attribute(embeds, sliding_window_shape, target = target, additional_forward_args = model,  **kwargs)
    else:
        if token_types:
            attr = exp.attribute(embeds, target = target, additional_forward_args = (model, batch.token_type_ids),  **kwargs)
        else:
            attr = exp.attribute(embeds, target = target, additional_forward_args = model,  **kwargs)

    return {
            'L2': summarize(attr, 'L2'),
            'L1': summarize(attr, 'L1'),
            'sum': summarize(attr, 'sum')
            }


def explain_model(loader, model, run_config,  attr_configs, origin, cuda):
    token_types = run_config.get('token_types', False)
    num_labels = run_config['num_labels']
    model.eval()
    if cuda:
        model.cuda()
    with tqdm(total=len(loader.batch_sampler)) as pbar:
        pbar.set_description('Compute attributions')
        explain_run = []
        for i,batch in enumerate(loader):
            if cuda:
                batch.cuda()
            with torch.no_grad():
                logits = compute_logits(model, batch, token_types).squeeze()
            for attr_class in range(num_labels):
                row = {'instance_id': batch.instance.cpu().item(),
                       'label': batch.labels.cpu().item(),
                       'pred': logits.argmax().cpu().item(),
                       'attr_class': attr_class,
                       'attr_class_pred_prob': torch.nn.functional.softmax(logits, dim = 0).cpu().numpy()[attr_class]}
                attributions = defaultdict(list)
                for attribution_method in attr_configs.keys():
                    kwargs =  attr_configs.get(attribution_method) or {}
                    kwargs = kwargs.copy()
                    if kwargs.get("baselines", False):
                        baseline = get_baseline(model, batch)
                        kwargs["baselines"] = baseline
                    attr = explain_batch(attribution_method, model, token_types, batch, target = attr_class, **kwargs)
                    attributions[attribution_method] = attr
                row.update(attributions)
                explain_run.append(row)
            batch.cpu()
            pbar.update(1)

        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        df = pd.DataFrame.from_records(explain_run)
    return df
