import json
import pickle

def to_json(d, path):
    with open(path, 'w') as f:
        json.dump(d, f)

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_json(path):
    with open(path, 'r') as f:
        d =  json.load(f)
    return d

def load_pickle(path):
    with open(path, 'rb') as f:
        obj =  pickle.load(f)
    return obj
