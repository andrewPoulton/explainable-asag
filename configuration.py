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


def load(*config_ids):
    master_config = load_configs_from_file(os.path.join('configs', 'main.yml'))
    config = dict()
    for R in ['DATA', 'DEFAULT']:
        config.update(master_config[R])
    for c in config_ids:
        print(f"Load configs for '{c}'.")
        config.update(master_config[c])
    config.update({'train_data': os.path.join(config['data_dir'], config['train_data_file']),
                   'val_data': os.path.join(config['data_dir'], config['val_data_file'])})
    return SimpleNamespace(**config)
