import torch
import transformers
import captum.attr as attributions

def update_token_type_embeddings(module, embedding_size, num_token_types = 4):
    for attr_str in dir(module):
        if attr_str == "token_type_embeddings":
            old_embeddings = module.__getattr__(attr_str)
            new_embeddings = torch.nn.Embedding(num_token_types, embedding_size)
            setattr(module, attr_str, new_embeddings)
            print(f"Updated token_type_embedding from {old_embeddings} to {new_embeddings}.")
            return

    for n, ch in module.named_children():
        embeds = update_token_type_embeddings(ch, embedding_size, num_token_types)

def load_model_from_disk(path):
    weights, config = torch.load(path, map_location='cpu')
    config = config['_items']
    mdl = transformers.AutoModelForSequenceClassification.from_pretrained(config['model_name'])
    if config.get('token_types', False):
        token_key = [k for k in weights.keys() if 'token_type' in k]
        assert len(token_key) == 1, f'Error, there are multiple keys that look like token type embeddings {token_key}'
        num_token_types, embedding_size = weights[token_key.pop()].shape
        update_token_type_embeddings(mdl, embedding_size, num_token_types)
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

def get_embeds(model, inputs):
    return get_word_embeddings(model)(inputs)

def get_baseline(model, batch):
    baseline_inputs = torch.where(batch.token_type_ids.eq(3), torch.zeros_like(batch.input), batch.input)
    return get_embeds(model, baseline_inputs)


def explainer(model, attribution_method = "IntegratedGradients"):
    attribution_method = attributions.__dict__[attribution_method]
    def func(embeds, model):
        return model(inputs_embeds = embeds).logits
    return attribution_method(func)

# def rank_tokens_by_attribution(batch, attributes, norm = 2, **kwargs):
#     norm = kwargs.get("norm", norm)
#     rank_order = attributes.norm(norm, dim = -1).argsort().squeeze()
#     ordered_tokens = batch.input.cpu().squeeze()[rank_order].numpy()[::-1].tolist()
#     return rank_order, ordered_tokens

def summarize(attr, aggr = 'norm', norm = 2):
    if aggr == 'norm':
        attr =  attr.norm(norm, dim = -1).squeeze(0)
    else:
        attr = attr.sum(dim=-1).squeeze(0)
    return attr.cpu().detach().numpy()


def explain_batch(attribution_method, model, batch, target = False, **kwargs):
    embeds = get_embeds(model, batch.input)
    with torch.no_grad():
        logits = model(inputs_embeds = embeds).logits.cpu().squeeze()
        pred = logits.argmax().item()
        pred_prob = torch.nn.functional.softmax(logits, dim=0).max().item()
    exp =  explainer(model, attribution_method)
    if kwargs.get("baselines", False):
        baseline = get_baseline(model, batch)
        kwargs["baselines"] = baseline
    if not target:
        target = pred
    if attribution_method == 'Occlusion':
        sliding_window_shape = (1,embeds.shape[-1])
        attr = exp.attribute(embeds, sliding_window_shape, target = target, additional_forward_args = model,  **kwargs)
    else:
        attr = exp.attribute(embeds, target = target, additional_forward_args = model,  **kwargs)
    return attr, pred, pred_prob
