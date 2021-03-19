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
__EXPLANATIONS_DIR__ = 'explain'

def explain(data_file, run_id, attribution_method, val_data_origin = 'unseen_answers'):
    try:
        for file in os.scandir(run_id):
            if file.name.endswith('.pt'):
                model_path = os.path.join(run_id, file.name)
                model, config  = load_model_from_disk(model_path)
    except:
        print('COULD NOT LOAD MODEL FROM DIR:', run_id)
        try:
            model, config = get_model_from_run_id(run_id, remove = False, check_exists = False)
        except:
            print('COULD NOT DOWNLOAD MODEL FROM WANDB WITH RUN_ID:', run_id)
            return None
    token_types = config.get('token_types', False)
    model.eval()
    if __CUDA__:
        model.cuda()
    kwargs = load_configs_from_file(os.path.join('configs','explain.yml'))["EXPLAIN"].get(attribution_method, {}) or {}
    if __CUDA__:
        num_workers = 8
    else:
        num_workers = 0
    expl_dataloader = dataset.dataloader(
        val_mode = True,
        data_file = data_file,
        data_source = config['data_source'],
        data_val_origin = val_data_origin,
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
        for batch in expl_dataloader:
            if __CUDA__:
                batch.cuda()
            for target in range(config['num_labels']):
                explanation_row = explain_batch(attribution_method, model, token_types, batch, target = target, **kwargs)
                explain_info.append(explanation_row)
            batch.cpu()
            pbar.update(1)
    df_expl = pd.DataFrame.from_records(records)
    df['model_path'] = config['model_name']
    df['source'] = config['data_source']
    df['attribution_method'] =  attribution_method
    df['run_id'] =  run_id
    df['num_labels'] = config['num_labels']
    file_name =  config['group'] + "_" + config['name']  '_' + run_id  + '_' + attribution_method + '.pkl'
    expl.to_pickle(os.path.join(__EXPLANATIONS_DIR__, file_name))
    model.cpu()
    #return expl

if __name__=='__main__':
    fire.Fire(explain)
