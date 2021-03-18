import os
import pandas as pd
import numpy as np
import configuration
import transformers
import torch
from sklearn.metrics import average_precision_score
from configuration import load_configs_from_file
from wandbinteraction import get_model_from_run_id
from dataset import dataloader
from scipy.stats import spearmanr
from collections import defaultdict
import re
from functools import partial
from itertools import chain, groupby
from collections import Counter
from dataset import SemEvalDataset, dataloader
from sklearn.preprocessing import MinMaxScaler

DATA_FILE = 'data/flat_semeval5way_test.csv'
ATTRIBUTION_DIR = 'attributions'
ANNOTATION_DIR = 'annotator/annotations'


### Human Agreement

def get_data_id_index(source):
    if source =='beetle':
        return range(468)
    if source =='scientsbank':
        return range(1315,1854)

def get_model_path(model):
    model_path = load_configs_from_file(os.path.join('configs', 'models.yml'))[model]['model_name']
    return model_path

def get_tokenizer(model):
    model_path = get_model_path(model)
    tokenizer =  transformers.AutoTokenizer.from_pretrained(model_path, lowercase=True)
    return tokenizer

def read_annotation(annotation_dir, annotation_file_name):
    annotator, source, origin1, origin2, data_id = annotation_file_name.split('.')[0].split('_')
    origin = '_'.join([origin1,origin2])
    with open(os.path.join(annotation_dir, annotation_file_name), 'r') as f:
        annotation = f.readline()[:-1]
    return {'annotator': annotator, 'source': source, 'origin': origin, 'data_id': data_id, 'annotation': annotation}

def get_annotations(annotation_dir):
    df = pd.DataFrame(columns = ['annotator','source','origin','annotation', 'data_id'])
    for annotation_file in os.scandir(annotation_dir):
        annotation_row = read_annotation(annotation_dir, annotation_file.name)
        df = df.append(annotation_row, ignore_index = True)
    return df

def aggregate_annotators(df_annotations):
    ann_aggr = lambda anns: [a for a, i in Counter(chain(anns)).items() if i > 1]
    df = pd.DataFrame(columns = df_annotations.columns)
    for data_id in df_annotations['data_id'].unique():
        _df = df_annotations[df_annotations['data_id'] == data_id]
        _row = _df.iloc[0]
        _row['annotator'] = 'aggr'
        _row['annotation'] = ann_aggr(_df['annotation'])
        df = df.append(_row, ignore_index = True)
    return df

def read_attribution(attribution_dir, attribution_file_name, attr_is_pred = False):
    print('Read attribution file', attribution_file_name)
    model, source, run_id, attribution_method = attribution_file_name.split('.')[0].split('_')
    df = pd.read_pickle(os.path.join(attribution_dir,attribution_file_name))
    if attr_is_pred:
        df = df[df['attr_class']== df['pred']]
        df['data_id'] = get_data_id_index(source)
    else:
        df['data_id'] = [i for i in get_data_id_index(source) for _ in range(df['attr_class'].max())]
    return {'df': df, 'model': model, 'source': source, 'run_id': run_id, 'attribution_method' : attribution_method}


def list_word_token_idx(text, tokenizer, special_tokens = False):
    """Returns  list of form [[0,1], [2], [3,4]] where sublist corresponds to words in text and indices to the positions of the word-piece tokens after encoding"""
    word_encodings =  [tokenizer.encode(word, add_special_tokens = special_tokens) for word in text.split()]
    b = tokenizer.encode_plus(text, add_special_tokens = special_tokens)
    c = [b.char_to_token(j) for j in range(len(text))] + [None]
    spc_pos = [i for i,t in enumerate(c) if t == None]
    word_idx = [list(set(c[i+1:j])) for i,j in zip([-1]+spc_pos, spc_pos)]
    return word_idx

def scale_to_unit_interval(attr):
    return torch.Tensor(MinMaxScaler().fit_transform(attr.unsqueeze(0)).squeeze(0)).long()


def compute_golden_saliency_vector(annotation, sentence):
    return np.array([1 if f'word_{k}' in annotation.split(',') else 0 for k in range(len( sentence.split()))])

def compute_student_answer_word_saliency_vector(attr, data_row, tokenizer):
    # We need to know the indices for student answer tokens and how they group as words
    _attr = torch.Tensor(attr).long()
    _attr = scale_to_unit_interval(_attr)
    assert _attr.shape == data_row['token_types'].shape, 'Attribution not same shape as encoded data.'
    _attr = torch.masked_select(_attr, data_row['token_types'].eq(3))
    _attr = _attr.index_select(0, torch.arange(0, _attr.size(0) - 1, dtype = int))
    w_idx = list_word_token_idx(data_row['student_answers'], tokenizer)
    assert _attr.size(0) == max(list(chain(*w_idx)) )+1, 'Student tokens to words mismatch'
    _attr = [np.max(_attr.numpy()[w]) for w in w_idx]
    return _attr


def get_testdata():
    df = pd.read_csv(DATA_FILE)
    df['data_id'] = df.index
    return df

def get_testdataset(model, source):
    dataset = SemEvalDataset(data_file = DATA_FILE, vocab_file = get_model_path(model))
    dataset.set_data_source(source)
    dataset.to_val_mode(source, 'answer')
    print(f"Test dataset loaded from {DATA_FILE} with {dataset.data.shape[0]} lines.")
    return dataset

def get_testdataloader(model, source):
    if torch.cuda.is_available():
        num_workers = 8
    else:
        num_workers = 0
    loader = dataloader(
        val_mode = True,
        data_file = DATA_FILE,
        data_source = source,
        vocab_file = get_model_path(model),
        num_labels = 2,
        train_percent = 100,
        batch_size = 1,
        drop_last = False,
        num_workers = num_workers)
    return loader


def compute_human_agreement(df_annotations, attribution_file_name, aggr = 'L2'):
    print('Computing human agreement (HA) for', attribution_file_name)
    attribution = read_attribution(ATTRIBUTION_DIR, attribution_file_name, attr_is_pred = True)
    loader = get_testdataloader(attribution['model'], attribution['source'])
    dataset = loader.dataset
    tokenizer = dataset.tokenizer
    df_anno = df_annotations[df_annotations['source'] == attribution['source']]
    df_attr = attribution['df'].set_index('data_id')
    AP = [0.0]*len(df_anno)
    for i in range(len(df_anno)):
        annotation_row = df_anno.iloc[i]
        data_id = int(annotation_row['data_id'])
        data_row = dataset.get_row(data_id)
        attribution_row = df_attr.loc[data_id]
        golden = compute_golden_saliency_vector(annotation_row['annotation'], data_row['student_answers'])
        saliency = compute_student_answer_word_saliency_vector(attribution_row['attr_' + aggr], data_row, tokenizer)
        ap = average_precision_score(golden, saliency)
        AP[i] = ap

    MAP = np.mean(AP)
    return {'attribution_file_name': attribution_file_name,
            'HA': MAP,
            'run_id': attribution['run_id'],
            'model': attribution['model'],
            'source': attribution['source'],
            'attribution_method' : attribution['attribution_method']}



### Rationale Consistency
###
### get the activations of layers is based on
### https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
### and
### https://github.com/copenlu/xai-benchmark/blob/master/saliency_eval/consist_data.py

def is_layer(name):
    layer_pattern = re.compile('^[a-z]*\.encoder\.layer.[0-9]*$')
    return layer_pattern.search(name) or name == 'classifier'

def save_activation(activations, name, mod, inp, out):
    # for encoder layers seems we get ([ tensor() ], )
    # while for classifier we get [tensor()]
    # so we select the corresponding te
    act = out
    while not isinstance(act, torch.Tensor) and len(act) == 1:
        act = act[0]
    activations[name] = act

def get_activations(model, **kwargs):
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


def compute_diff_activation(model1, model2, batch):
    activations1 = get_activations(model1, input_ids = batch.input, attention_mask = batch.generate_mask())
    activations2 = get_activations(model2, input_ids = batch.input, attention_mask = batch.generate_mask())
    keys = activations1.keys()
    assert  keys == activations2.keys()
    act_diff = np.mean([(activations1[key]- activations2[key]).abs().mean().cpu().item() for key in keys])
    print(act_diff)
    return act_diff

def compute_diff_attribution(attr1, attr2):
    return np.mean(np.abs(np.array(attr1) -np.array(attr2)))


def compute_rationale_consistency(attribution_file1, attribution_file2, aggr = 'L2', **kwargs):
    A1 = read_attribution(ATTRIBUTION_DIR, attribution_file1, attr_is_pred=True)
    A2 = read_attribution(ATTRIBUTION_DIR, attribution_file2, attr_is_pred=True)
    assert A1['model'] == A2['model'] and A1['source'] == A2['source'] and A1['attribution_method'] == A2['attribution_method']
    model1, config1 = get_model_from_run_id(A1['run_id'], **kwargs)
    model2, config2 = get_model_from_run_id(A2['run_id'], **kwargs)
    model1.eval()
    model2.eval()
    assert config1['num_labels'] == config2['num_labels']
    config = config1
    loader = dataloader(
        val_mode = True,
        data_file = DATA_FILE,
        data_source = config['data_source'],
        vocab_file = config['model_name'],
        num_labels = config['num_labels'],
        train_percent = 100,
        batch_size = 1,
        drop_last = False,
        num_workers = 0)
    attrs1 = A1['df']['attr_' + aggr]
    attrs2 = A2['df']['attr_' + aggr]
    len_data = len(loader)
    assert len_data == len(attrs1) == len(attrs2)
    diff_activation = np.empty(len_data)
    diff_attribution = np.empty(len_data)
    for i, batch in enumerate(loader):
        attr1, attr2 = attrs1.iloc[i], attrs2.iloc[i]
        diff_activation[i] = compute_diff_activation(model1, model2, batch)
        diff_attribution[i] = compute_diff_attribution(attr1, attr2)
    r = spearmanr(diff_activation, diff_attribution)
    return r
