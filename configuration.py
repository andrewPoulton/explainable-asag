CONFIGS_DIR = 'configs'
REQUIRED  = ['DATA', 'DEFAULT']

from warnings import warn
from types import SimpleNamespace
import os
import yaml

def load_configs_from_file(file_path):
    configs = dict()
    with open(file_path, 'r') as f:
        y = yaml.load_all(f, Loader = yaml.FullLoader)
        for d in y:
            configs.update(d)
    return configs


def load_all_configs():
    master_config = dict()
    for file_name in os.listdir(CONFIGS_DIR):
        configs = load_configs_from_file(os.path.join(CONFIGS_DIR, file_name))
        master_config.update(configs)
    return master_config


def load(*config_ids):
    master_config = load_all_configs()
    config = dict()
    for R in REQUIRED:
        config.update(master_config[R])
        valid_config_ids.remove(R)
    for c in config_ids:
        print(f"Load configs for '{c}'.")
        config.update(master_config[c])
    config.update({'train_data': os.path.join(config['data_dir'], config['train_data_file']),
                   'val_data': os.path.join(config['data_dir'], config['val_data_file'])})
    return SimpleNamespace(**config)
