import torch
import transformers
from collections import defaultdict
import re
from functools import partial

def load_model_from_disk(path):
    weights, config = torch.load(path, map_location='cpu')
    config = config['_items']
    mdl = transformers.AutoModelForSequenceClassification.from_pretrained(config['model_name'])
    if config.get('token_types', False):
        token_key = [k for k in weights.keys() if 'token_type' in k]
        assert len(token_key) == 1, f'Error, there are multiple keys that look like token type embeddings {token_key}'
        num_token_types, embedding_size = weights[token_key.pop()].shape
        update_token_type_embeddings(mdl, embedding_size, None, num_token_types)
    mdl.load_state_dict(weights)
    print(f"Loaded model {config['model_name']} from {path} successfully.")
    return mdl, config


def update_token_type_embeddings(module, embedding_size, initializer_range, num_token_types = 4):
    for attr_str in dir(module):
        if attr_str == "token_type_embeddings":
            old_embeddings = module.__getattr__(attr_str)
            new_embeddings = torch.nn.Embedding(num_token_types, embedding_size)
            if initializer_range:
                new_embeddings.weight.data.normal_(mean = 0.0, std = initializer_range)
            setattr(module, attr_str, new_embeddings)
            print(f"Updated token_type_embedding from {old_embeddings} to {new_embeddings}.")
            return

    for n, ch in module.named_children():
        embeds = update_token_type_embeddings(ch, embedding_size, initializer_range, num_token_types)


def get_word_embeddings(module):
    for attr_str in dir(module):
        if attr_str == "word_embeddings":
            return  getattr(module, attr_str)

    for n, ch in module.named_children():
        embeds = get_word_embeddings(ch)
        if embeds:
            return embeds


### get the activations of layers is based on
### https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
### and
### https://github.com/copenlu/xai-benchmark/blob/master/saliency_eval/consist_data.py

def is_layer(name):
    layer_pattern = re.compile('^[a-z]*.encoder.layer.[0-9]*$')
    return bool(layer_pattern.search(name) or name == 'classifier')

def save_activation(activations, name, mod, inp, out):
    act = out
    # for encoder layers seems we get ([ tensor() ], )
    # while for classifier we get [tensor()]
    # so we select the corresponding tensor to save
    while not isinstance(act, torch.Tensor) and len(act) == 1:
        act = act[0]
    activations[name] = act

def get_layer_activations(model, **kwargs):
    activations = defaultdict(torch.Tensor)
    handles = []
    for name, module in model.named_modules():
        if is_layer(name):
            handle = module.register_forward_hook(partial(save_activation, activations, name))
            handles.append(handle)

    with torch.no_grad():
        model(**kwargs)

    # is this needed?
    for handle in handles:
        handle.remove()

    return activations
