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
import os

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
    token_types = config.get('token_types', False)
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
    # NOTE: This only works for batch_size = 1 and relies on it for now
    with tqdm(total=len(expl_dataloader.batch_sampler)) as pbar:
        pbar.set_description(f'Compute {attribution_method}:')
        explain_info = []
        model_info = []
        for batch in expl_dataloader:
            if __CUDA__:
                batch.cuda()
            for target in range(config['num_labels']):
                explanation_row = explain_batch(attribution_method, model, token_types, batch, target = target, **kwargs)
                explain_info.append(explanation_row)
                model_info.append({'model_path': config['model_name'],
                                   'source': config['data_source'],
                                   'attribution_method': attribution_method,
                                   'run_id': model_dir})
            batch.cpu()
            pbar.update(1)
    expl = pd.DataFrame.from_records(records)
    file_name =  config['data_source'] + "_" + config['name']
    if config['token_types']:
        file_name += '=token-types'
    file_name +=  '_' + model_dir  + '_' + attribution_method + '.pkl'
    expl.to_pickle(os.path.join('explained', file_name))
    model.cpu()
    #return expl


if __name__=='__main__':
    fire.Fire(explain)
