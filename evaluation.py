import os
import re
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
from functools import partial
from itertools import chain, groupby
from collections import Counter
from dataset import SemEvalDataset, dataloader
from sklearn.preprocessing import MinMaxScaler
from dataset import __TEST_DATA__
from tqdm import tqdm

class AnnotationData:
    def __init__(self, annotation_dir):
        self.annotation_dir = annotation_dir
        self.annotations = self.load_annotations()
        self.annotator_names = self.annotations['annotator'].unique().tolist()
        self.annotations['annotator_label'] = self.annotations['annotator'].replace({n:i+1 for i,n in enumerate(self.annotator_names)})
        self.aggr_annotator = 'aggr'
        self.aggr_function = lambda anns: [a for a, i in Counter(chain(anns)).items() if i > 1]
        self.append_aggregated_annotations()
        self._annotations = self.annotations.copy()

    def load_annotations(self):
        annotation_list = []
        annotation_files = [f.name for f in os.scandir(self.annotation_dir) if f.name.endswith('annotation')]
        for annotation_file_name in tqdm(annotation_files, desc=f"Loading annotations from {os.path.abspath(self.annotation_dir)}"):
            annotator, source, origin1, origin2, instance_id = os.path.splitext(annotation_file_name)[0].split('_')
            origin = '_'.join([origin1,origin2])
            with open(os.path.join(self.annotation_dir, annotation_file_name), 'r') as f:
                annotation = f.readline()[:-1]
                annotation_list.append({'instance_id': instance_id,'source': source, 'origin': origin, 'annotator': annotator,  'annotation': annotation})
        return pd.DataFrame.from_records(annotation_list)

    def append_aggregated_annotations(self):
        for instance_id in tqdm(self.annotations['instance_id'].unique() , desc = 'Calculate aggregated annotations'):
            _df = self.annotations[self.annotations['instance_id'] == instance_id]
            _row = _df.iloc[0].copy()
            _row['annotator'] = self.aggr_annotator
            _row['annotation'] = self.aggr_function(_df['annotation'])
            _row['annotator_label'] = 0
            self.annotations = self.annotations.append(_row, ignore_index = True)

    def get_golden_saliency(self, annotator, instance_id, data):
        # should throw error if annotator has not annotated specific instance
        from_annotator =  self.annotations[self.annotations['annotator'] == annotator].set_index('instance_id')
        annotation = from_annotator.loc[instance_id]
        student_answer = data.get_instance(instance_id)['student_answers']
        return [1 if f'word_{k}' in annotation.split(',')
                                else 0 for k,w in student_answer.split())])



# def read_attribution(attribution_dir, attribution_file_name, attr_is_pred = False):
#     print('Read attribution file', attribution_file_name)
#     model, source, run_id, attribution_method = attribution_file_name.split('.')[:-1].split('_')
#     df = pd.read_pickle(os.path.join(attribution_dir,attribution_file_name))
#     if attr_is_pred:
#         df = df[df['attr_class']== df['pred']]
#         df['instance_id'] = get_instance_id_index(source)
#     else:
#         df['instance_id'] = [i for i in get_instance_id_index(source) for _ in range(df['attr_class'].max())]
#     return {'df': df, 'model': model, 'source': source, 'run_id': run_id, 'attribution_method' : attribution_method}


# def list_word_token_idx(text, tokenizer, special_tokens = False):
#     """Returns  list of form [[0,1], [2], [3,4]] where sublist corresponds to words in text and indices to the positions of the word-piece tokens after encoding"""
#     word_encodings =  [tokenizer.encode(word, add_special_tokens = special_tokens) for word in text.split()]
#     b = tokenizer.encode_plus(text, add_special_tokens = special_tokens)
#     c = [b.char_to_token(j) for j in range(len(text))] + [None]
#     spc_pos = [i for i,t in enumerate(c) if t == None]
#     word_idx = [list(set(c[i+1:j])) for i,j in zip([-1]+spc_pos, spc_pos)]
#     return word_idx

def scale_to_unit_interval(attr):
    return torch.Tensor(MinMaxScaler().fit_transform(torch.Tensor(attr).unsqueeze(0)).squeeze(0)).long()

def compute_golden_saliency(annotation, sentence):
    return [1 if f'word_{k}' in annotation.split(',') else 0 for k,w in enumerate(sentence.split())]

def compute_attribution_saliency(attr, data):
    # We need to know the indices for student answer tokens and how they group as words
    row = data.get_instance(attr['instance'])
    return _attr.tolist()


def compute_human_agreement(dataset, attribution_df, aggr):
    print('Computing human agreement (HA) for', attribution_file_name)
    attribution = read_attribution(ATTRIBUTION_DIR, attribution_file_name, attr_is_pred = True)
    dataset = get_testdataset(attribution['model'], attribution['source'])
    tokenizer = dataset.tokenizer
    df_anno = df_annotations[df_annotations['source'] == attribution['source']]
    df_attr = attribution['df'].set_index('instance_id')
    AP = [0.0]*len(df_anno)
    for i in range(len(df_anno)):
        annotation_row = df_anno.iloc[i]
        instance_id = int(annotation_row['instance_id'])
        data_row = dataset.get_row(instance_id)
        attribution_row = df_attr.loc[instance_id]
        golden = compute_golden_saliency_vector(annotation_row['annotation'], data_row['student_answers'])
        saliency = compute_student_answer_word_saliency_vector(attribution_row['attr_' + aggr], data_row, tokenizer)
        ap = average_precision_score(golden, saliency)
        AP[i] = ap

    MAP = np.mean(AP)
    return {'attribution_file_name': attribution_file_name,
            'metric': MAP,
            'run_id': attribution['run_id'],
            'model': attribution['model'],
            'source': attribution['source'],
            'attribution_method' : attribution['attribution_method'],
            'aggr': aggr}



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


def compute_rationale_consistency(attribution_file1, attribution_file2, aggr, **kwargs):
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
    return {'attribution_file_name1': attribution_file1.name,
            'attribution_file_name2': attribution_file2.name,
            'run_id1': A1['run_id'],
            'run_id2': A2['run_id'],
            'metric': r,
            'model': A1['model'],
            'source': A1['source'],
            'attribution_method' : A1['attribution_method'],
            'aggr': aggr}
