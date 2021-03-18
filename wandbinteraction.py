# the goal of this script is to download and save wandb stats
# stolen straight from https://docs.wandb.ai/library/public-api-guide

import wandb
import numpy as np
import pandas as pd
import os
import shutil
from explanation import load_model_from_disk

def save_experiments_info():
    api = wandb.Api()
    runs = api.runs("sebaseliens/explainable-asag")
    my_stats_list = []
    summary_list = []
    config_list = []
    for run in runs:
        # add some stats I want
        max_f1 = 0.0
        for i, row in run.history(keys = ['f1']).iterrows():
            if row['f1'] > max_f1:
                max_f1 = row['f1']

        my_stats_list.append({'run_id':run.id, 'max_f1': max_f1})

        # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # run.config is the input metrics.  We remove special values that start with _.
        config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')})

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    my_stats_df = pd.DataFrame.from_records(my_stats_list)
    df = pd.concat([my_stats_df, config_df, summary_df], axis=1)
    df.to_csv("results/experiments.csv")
    pass

def download_run(run_id, ext = None):
    api = wandb.Api()
    run = api.run("sebaseliens/explainable-asag/" + run_id)
    file_names = []
    print('Downloading run', run_id)
    for f in run.files():
        if ext:
            if f.name.endswith(ext):
                f.download(run_id, replace = True)
                file_names.append(f.name)
                break
            else:
                continue
        else:
            f.download(run_id, replace = replace)
            file_names.append(f.name)
    print('Downloaded:', *file_names)
    return file_names

def remove_run(run_id):
    api = wandb.Api()
    runs = api.runs("sebaseliens/explainable-asag")
    if run_id in [run.id for run in runs]:
        shutil.rmtree(run_id)
    else:
        print(f'Did not remove {run_id}. It is not a run.')
    pass


def get_model_from_run_id(run_id, remove = True, check_exists = False):
    path_to_model = None
    if check_exists:
        if os.path.exists(run_id):
            for f in os.scandir(run_id):
                if f.name.endswith('.pt'):
                    path_to_model =  os.path.join(run_id, f.name)
                    break
    if not path_to_model or not check_exists:
        file_names = download_run(run_id, ext = '.pt')
        path_to_model = os.path.join(run_id, file_names[0])

    mdl, config = load_model_from_disk(path_to_model)
    if remove:
        remove_run(run_id)
    return mdl, config

def get_wandb_df():
    wandb_df = pd.read_csv('results/experiments.csv', index_col = 0)
    return wandb_df

def get_group(source, token_types = False):
    wandb_df = get_wandb_df()
    if source == 'scientsbank':
        group = 'group'
    elif source =='beetle':
        group = 'beetlebeetle'
    else:
        error('Only allowed data sources: scientsbank and beetle')
    return group

def runs_info(source, drop_large = True, token_types = False, print_out = False):
    df_wandb = get_wandb_df()
    group = get_group(source, token_types = token_types)
    dfs = [df_wandb[df_wandb['group'] == group + j][['run_id', 'name', 'token_types']] for j in ['-1','-2','-3']]
    for i, _df in enumerate(dfs):
        _df['group'] = i+1
    df = pd.concat(dfs, axis = 0)
    df['token_types'].fillna(False, inplace = True)
    df['source'] = source
    df.reset_index(drop = True, inplace = True)
    if drop_large:
        df = df[~df['name'].str.contains('large')]
    if print_out:
        for i, row in df.iterrows():
            print(row.name, '\t#', f"{row['name']} {row['source']}-{row['group']}")
    return df
