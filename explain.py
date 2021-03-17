import fire
import os
import pandas as pd
import torch
from configuration import load_configs_from_file
import dataset
from tqdm import tqdm
from explanation import (
    load_model_from_disk,
    explain_batch,
    summarize
)
from wandbinteraction import get_model_from_run_id

__CUDA__ = torch.cuda.is_available()

def explain(data_file, model_dir, attribution_method):
    try:
        for file in os.scandir(model_dir):
            if file.name.endswith('.pt'):
                model_path = os.path.join(model_dir, file.name)
                model, config  = load_model_from_disk(model_path)
    except:
        print('COULD NOT LOAD MODEL FROM DIR:', model_dir)
        try:
            model, config = get_model_from_run_id(model_dir, remove = False, check_exists = False)
        except:
            print('COULD NOT DOWNLOAD MODEL FROM WANDB WITH RUN_ID:', model_dir)
            return None

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
