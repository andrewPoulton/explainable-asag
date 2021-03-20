import torch
import transformers

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


def get_word_embeddings(module):
    for attr_str in dir(module):
        if attr_str == "word_embeddings":
            return  getattr(module, attr_str)

    for n, ch in module.named_children():
        embeds = get_word_embeddings(ch)
        if embeds:
            return embeds
