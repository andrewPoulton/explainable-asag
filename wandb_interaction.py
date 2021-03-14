# the goal of this script is to download and save wandb stats
# stolen straight from https://docs.wandb.ai/library/public-api-guide

import wandb
import numpy as np
import os
import shutil
from explain import load_model_from_disk

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

    import pandas as pd
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    my_stats_df = pd.DataFrame.from_records(my_stats_list)
    df = pd.concat([my_stats_df, config_df, summary_df], axis=1)

    df.to_csv("tables/experiments.csv")
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
            f.download(run_id, replace = True)
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


def get_model_from_run_id(run_id):
    file_names = download_run(run_id, ext = '.pt')
    path_to_model = os.path.join(run_id, file_names[0])
    mdl, config = load_model_from_disk(path_to_model)
    remove_run(run_id)
    return mdl, config
