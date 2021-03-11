# the goal of this script is to download and save wandb stats
# stolen straight from https://docs.wandb.ai/library/public-api-guide

import wandb
import numpy as np
api = wandb.Api()

# Change oreilly-class/cifar to <entity/project-name>
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
