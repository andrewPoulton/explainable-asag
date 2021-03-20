# the goal of this script is to download and save wandb stats
# stolen straight from https://docs.wandb.ai/library/public-api-guide
import wandb
import numpy as np
import pandas as pd
import os
import shutil
import torch
from modelhandling import load_model_from_disk

__api__ = wandb.Api()
__runs__ = __api__.runs("sebaseliens/explainable-asag")

def download_run(run_id, ext = None):
    run = __api__.run("sebaseliens/explainable-asag/" + run_id)
    file_names = []
    for f in run.files():
        if ext:
            if f.name.endswith(ext):
                print('Downloading:', os.path.join(run_id, f.name))
                f.download(run_id, replace = True)
                file_names.append(f.name)
                break
            else:
                continue
        else:
            f.download(run_id, replace = replace)
            file_names.append(f.name)
    return file_names

def remove_run(run_id):
    if run_id in [run.id for run in __runs__]:
        shutil.rmtree(run_id)
        print('Deleted:', run_id)
    else:
        print(f'Did not remove {run_id}. It is not a run.')
    pass

def load_model_from_run_id(run_id, remove = True, check_exists = False):
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

def download_experiment_info():
    my_stats_list = []
    summary_list = []
    config_list = []
    for run in __runs__:
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
    return df

def get_run_ids(*groups):
    return [run.id for run in __runs__ if run.config['group'] in groups or not groups]
