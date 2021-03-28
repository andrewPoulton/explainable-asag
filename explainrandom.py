import fire
import os
import pandas as pd
import torch
from configuration import load_configs_from_file
from dataset import dataloader, __TEST_DATA__
from explanation import explain_model
from wandbinteraction import get_runs, load_model_from_run
import transformers

__CUDA__ = torch.cuda.is_available()
__EXPLANATIONS_DIR__ = 'explained'
__num_workers__ = 4

def explain(source, name, token_types = False, origin = 'unseen_answers'):
    num_labels = 2
    group =  source + ('-token_types' if token_types else '')
    attr_directory = os.path.join(__EXPLANATIONS_DIR__, 'random_'+ group)
    os.makedirs(attr_directory, exist_ok=True)
    path_to_attribution_file = os.path.join(attr_directory, name + '.pkl')
    model_path = load_configs_from_file(os.path.join('configs', 'main.yml'))[name]['model_name']
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = num_labels)
    config = {
        'token_types' : token_types,
        'num_labels' : num_labels
    }
    model.init_weights()
    loader = dataloader(
        data_file = __TEST_DATA__,
        val_mode = True,
        data_source = source,
        data_val_origin = origin,
        vocab_file = model_path,
        num_labels = num_labels,
        train_percent = 100,
        batch_size = 1,
        drop_last = False,
        num_workers = __num_workers__ if __CUDA__ else 0)
    attr_configs = load_configs_from_file(os.path.join('configs','explain.yml'))['EXPLAIN']
    if 'large' in name:
        for attribution_method in attr_configs.keys():
            if attr_configs[attribution_method] and 'internal_batch_size' in attr_configs[attribution_method]:
                attr_configs[attribution_method]['internal_batch_size'] = attr_configs[attribution_method]['internal_batch_size']//2
    df = explain_model(loader, model, config,  attr_configs, origin, __CUDA__)
    df['run_id'] = 'random'
    df['model'] = name
    df['model_path'] = model_path
    df['source'] = source
    df['origin'] = origin
    df['num_labels'] = num_labels
    df['group'] = group
    df['token_types'] = token_types
    df.to_pickle(path_to_attribution_file)

if __name__=='__main__':
    fire.Fire(explain)
