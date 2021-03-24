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
 __num_workers__ = 2

def explain(*wandb_groups, origin = 'unseen_answers'):
    if not os.path.isdir(__EXPLANATIONS_DIR__):
        os.mkdir(__EXPLANATIONS_DIR__)
    for run in get_runs(*wandb_groups):
        attr_directory = os.path.join(__EXPLANATIONS_DIR__, run.config['group'])
        if not os.path.isdir(attr_directory):
            os.mkdir(attr_directory)
        path_to_attribution_file = os.path.join(attr_directory, run.id + '.pkl')
        if os.path.isfile(os.path.join(__EXPLANATIONS_DIR__, run.config['group'], run.id + '.pkl')):
            print('Run already explained:', run.id)
            continue
        print('Explaining run:', run.id)
        try:
            model, cfg = load_model_from_run(run)
        except:
            continue
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
            num_workers = __num_workers__ if __CUDA__ else 0)
        if not 'token_types' in  run.config:
            run.config.update({'token_types': False})
        attr_configs = load_configs_from_file(os.path.join('configs','explain.yml'))['EXPLAIN']
        if 'large' in run.config['model_name']:
            for attribution_method in attr_configs.keys():
                if attr_configs[attribution_method] and 'internal_batch_size' in attr_configs[attribution_method]:
                    attr_configs[attribution_method]['internal_batch_size'] = attr_configs[attribution_method]['internal_batch_size']//2
        df = explain_model(loader, model, run.config,  attr_configs, origin, __CUDA__)
        df['run_id'] = run.id
        df['model'] = run.config['name']
        df['model_path'] = run.config['model_name']
        df['source'] = run.config['data_source']
        df['origin'] = origin
        df['num_labels'] = run.config['num_labels']
        df['group'] = run.config['group']
        df['token_types'] = run.config['token_types']
        df.to_pickle(path_to_attribution_file)

if __name__=='__main__':
    fire.Fire(explain)
