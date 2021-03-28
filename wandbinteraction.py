# the goal of this script is to download and save wandb stats
# stolen straight from https://docs.wandb.ai/library/public-api-guide
import wandb
import numpy as np
import pandas as pd
import os
import shutil
import torch
from modelhandling import load_model_from_disk
import contextlib
from tempfile import mkdtemp

__api__ = wandb.Api()
__runs__ = __api__.runs("sebaseliens/explainable-asag")

def get_run_ids(*groups):
    return [run.id for run in __runs__ if run.config['group'] in groups or not groups]

def get_runs(*groups):
    return [run for run in __runs__ if run.config['group'] in groups or not groups]

def as_run(run):
    if isinstance(run, str):
        run = __api__.run("sebaseliens/explainable-asag/" + run)
    return run

def as_run_id(run):
    if not isinstance(run, str):
        run = run.id
    return run

@contextlib.contextmanager
def make_temp_dir():
    temp_dir = mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

def load_model_from_wandb(run):
    run = as_run(run)
    with make_temp_dir() as temp_dir:
        files = run.files()
        for f in files:
            if f.name.endswith('.pt'):
                print('Downloading model from WandB:', os.path.join(run.id, f.name))
                f.download(temp_dir, replace = True)
                path_to_model = os.path.join(temp_dir, f.name)
        model, config = load_model_from_disk(path_to_model)
    return model, config

def download_run(run,  ext = None):
    run = as_run(run)
    files = run.files()
    for f in files:
        if ext and not f.name.endswith(ext):
            continue
        else:
            try:
                print('Downloading:', os.path.join(run.id, f.name))
                f.download(run.id, replace = True)
            except:
                print('Download failed:', os.path.join(run.id, f.name))
                print(f'Removing dir {os.path.join(run.id,"")}')
                shutil.rmtree(run.id)


def delete_run(run):
    run = get_run_id(run)
    if run in [run.id for run in __runs__] and os.path.isdir(run):
        shutil.rmtree(run_id)
        print('Deleted directory:', os.path.join(run_id,''))
    else:
        print(f'Did not remove {run_id}. Not a run or not a directory.')
    pass

remove_run = delete_run

def load_model_from_run(run, **kwargs):
    run = as_run_id(run)
    try:
        for file_name in os.listdir(run):
            if file_name.endswith('.pt'):
                model, config = load_model_from_disk(os.path.join(run, file_name))
                return model, config
    except:
        model, config = load_model_from_wandb(run)
        return model, config

# maybe somewhere in repos still this... clean up
load_model_from_run_id = load_model_from_run


def get_run_info(run):
    run = as_run(run)
    return {'model': run.config['name'],
            'source': run.config['data_source'],
            'token_types': run.config.get('token_types', False)}
