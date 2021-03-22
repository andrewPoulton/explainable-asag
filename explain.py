import fire
import os
import pandas as pd
import torch
from configuration import load_configs_from_file
from dataset import dataloader, __TEST_DATA__
from explanation import explain_model
from wandbinteraction import get_runs, load_model_from_run

__CUDA__ = torch.cuda.is_available()
__EXPLANATIONS_DIR__ = 'explained'

def explain(*wandb_groups, origin = 'unseen_answers'):
    if not os.path.isdir(__EXPLANATIONS_DIR__):
        os.mkdir(__EXPLANATIONS_DIR__)
    for run in get_runs(*wandb_groups):
        if os.path.isfile(os.path.join(__EXPLANATIONS_DIR__, run.config['group'], run.id + '.pkl')):
            print('Run already explained:', run.id)
            continue
        print('Explaining run:', run.id)
        model, cfg = load_model_from_run(run)
        loader = dataloader(
            data_file = __TEST_DATA__,
            val_mode = True,
            data_source = run.config['data_source'],
            data_val_origin = origin,
            vocab_file = run.config['model_name'],
            num_labels = run.config['num_labels'],
            train_percent = 100,
            batch_size = 1,
            drop_last = False,
            num_workers = run.config['num_workers'] if __CUDA__ else 0)
        token_types = config.get('token_types', False)
        attr_methods = load_configs_from_file(os.path.join('configs','explain.yml'))['EXPLAIN'].keys()
        df = explain_model(loader, model, run.config,  attr_methods, origin, __CUDA__)
        df['run_id'] = run.id
        df['model'] = run.config['name']
        df['model_path'] = run.config['model_name']
        df['source'] = run.config['data_source']
        df['origin'] = origin
        df['num_labels'] = run.config['num_labels']
        df['group'] = run.config['group']
        df['token_types'] = run.config['token_types']

        if not os.path.isdir(os.path.join(__EXPLANATIONS_DIR__, run.config['group'])):
            os.mkdir(os.path.join(__EXPLANATIONS_DIR__, run.config['group']))
        df.to_pickle(os.path.join(__EXPLANATIONS_DIR__, run.config['group'], run.id + '.pkl'))


if __name__=='__main__':
    fire.Fire(explain)
