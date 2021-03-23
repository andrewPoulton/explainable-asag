import os
import re
import pandas as pd
import numpy as np
import configuration
import transformers
import torch
from sklearn.metrics import average_precision_score
from configuration import load_configs_from_file
from dataset import dataloader
from scipy.stats import spearmanr
from collections import defaultdict
from functools import partial
from itertools import chain, groupby
from collections import Counter
from dataset import SemEvalDataset, dataloader, __TEST_DATA__
from sklearn.preprocessing import MinMaxScaler
from dataset import SemEvalDataset, dataloader, pairdataloader
from tqdm import tqdm
from configuration import load_configs_from_file
from modelhandling import get_layer_activations
from wandbinteraction import load_model_from_run, remove_run
import warnings
import gc
from collections import defaultdict
from copy import deepcopy
from dataset import pad_tensor_batch
#warnings.filterwarnings("error")
#['IntegratedGradients', 'InputXGradient','Saliency','GradientShap','Occlusion']
__RESULTS_DIR__ = 'results'
__attr_methods__ = list(load_configs_from_file(os.path.join('configs', 'explain.yml'))['EXPLAIN'].keys())
__aggr__ = ['L2', 'L1', 'sum']

class AnnotationData:
    def __init__(self, annotation_dir, aggr = True):
        self.annotation_dir = annotation_dir
        self.load_annotations()
        if aggr:
            self.aggr_annotator = 'aggr'
            self.append_aggregated_annotations()
        self._annotations = self.annotations.copy()
        self.source = False
        self.current_annotators = False

    def load_annotations(self):
        annotation_files = [f.name for f in os.scandir(self.annotation_dir) if f.name.endswith('annotation')]
        annotations_list = []
        annotator_names = []
        annotator_index = 0
        for annotation_file_name in tqdm(annotation_files, desc=f"Loading annotations from {os.path.abspath(self.annotation_dir)}"):
            annotator, source, origin1, origin2, instance_id = os.path.splitext(annotation_file_name)[0].split('_')
            origin = '_'.join([origin1,origin2])
            if not annotator in annotator_names:
                annotator_names.append(annotator)
                annotator_index += 1
            with open(os.path.join(self.annotation_dir, annotation_file_name), 'r') as f:
                annotation = f.readline()[:-1]
                annotations_list.append({'instance_id': instance_id,'source': source, 'origin': origin,
                                         'annotator': annotator, 'annotator_index': annotator_index,
                                         'annotation': annotation})
        self.annotations = pd.DataFrame.from_records(annotations_list)
        self.annotations['instance_id'] = self.annotations['instance_id'].astype(int)
        self.annotator_names = annotator_names

    def append_aggregated_annotations(self):
        for instance_id in tqdm(self.annotations['instance_id'].unique() , desc = 'Calculate aggregated annotations'):
            _df = self.annotations[self.annotations['instance_id'] == instance_id]
            _row = _df.iloc[0].copy()
            _row['annotator'] = self.aggr_annotator
            _row['annotation'] = ann_aggr_function(_df['annotation'])
            _row['annotator_label'] = 0
            self.annotations = self.annotations.append(_row, ignore_index = True)

    def set_annotator(self, *annotators):
        if annotators:
            self.annotations = self._annotations[self._annotations['annotator'].isin(annotators)]
        else:
            self.annotations = self._annotations.copy()
        if self.source:
            self.annotations = self.annotations[self.annotations['source'] == self.source]
        self.current_annotators = annotators

    def set_source(self, source):
        if source:
            self.annotations = self._annotations[self._annotations['source'] == source]
        else:
            self.annotations = self._annotations.copy()
        if self.current_annotators:
            self.annotations = self.annotations[self.annotations['annotator'].isin(self.current_annotators)]
        self.source = source


class AttributionData:
    def __init__(self, attribution_file):
        self.df  = pd.read_pickle(attribution_file)
        if 'token_types' not in self.df.columns:
            self.df['token_types'] = False
        self.attr_class = None
        self.token_types = self.df['token_types'][0]
        self.run_id = self.df['run_id'].unique()[0]
        self.model_name = self.df['model'].unique()[0]
        self.model_path = self.df['model_path'].unique()[0]
        self.source = self.df['source'].unique()[0]
        self.num_labels = self.df['num_labels'].unique()[0]
        self.origin = self.df['origin'].unique()[0]
        self.group = self.df['group'].unique()[0]
        self._df = self.df.copy()

    def is_compatible(self, attr_data):
        return self.model_path == attr_data.model_path and \
                self.source == attr_data.source and \
                self.num_labels == attr_data.num_labels and \
                self.origin ==  attr_data.origin and \
                self.token_types == attr_data.token_types

    def get_dataloader(self):
        loader = dataloader(
            data_file = __TEST_DATA__,
            val_mode = True,
            data_source = self.source,
            data_val_origin = self.origin,
            vocab_file = self.model_path,
            num_labels = self.num_labels,
            train_percent = 100,
            batch_size = 1,
            drop_last = False,
            num_workers = 8 if  torch.cuda.is_available() else 0)
        return loader


    def get_pairdataloader(self, **kwargs):
        loader = pairdataloader(
            data_file = __TEST_DATA__,
            val_mode = True,
            data_source = self.source,
            data_val_origin = self.origin,
            vocab_file = self.model_path,
            num_labels = self.num_labels,
            train_percent = 100,
            num_workers = 8 if  torch.cuda.is_available() else 0,
            **kwargs)
        return loader

    def get_dataset(self):
        return self.get_dataloader().dataset

    def set_attr_class(self, target):
        if target == 'pred':
            self.df = self._df[self._df['attr_class'] == self._df['pred']]
        elif target in range(self.num_labels):
            self.df = self._df[self._df['attr_class'] == target]
        else:
            self.df = self._df.copy()
        self.attr_class = target

    def load_model(self):
        model, config = load_model_from_run(self.run_id)
        return model, config


def golden_saliency(annotation, instance_id, dataset):
    student_answer_words = dataset.get_instance(instance_id)['student_answers'].split()
    golden =  [1 if f'word_{k}' in annotation.split(',')
               else 0 for k,w in enumerate(student_answer_words)]
    return golden

def ann_aggr_function(anns):
    return ','.join([word_k for word_k, i in Counter(chain(anns)).items() if i > 1])

def scale_to_unit_interval(attr):
    _attr = MinMaxScaler().fit_transform([[a] for a in attr])
    return [a[0] for a in _attr]


def compute_human_agreement(attr_data, ann_data):
    ann_data.set_source(attr_data.source)
    if not attr_data.attr_class == 'pred':
        attr_data.set_attr_class('pred')
        df = attr_data.df.set_index('instance_id')
        ap_scores = []
    dataset = attr_data.get_dataset()
    for i, ann in tqdm(ann_data.annotations.iterrows(), desc = f'Computing human agreement: {attr_data.run_id}'):
        annotation = ann['annotation']
        instance_id = ann['instance_id']
        instance = dataset.get_instance(instance_id, word_structure = True)
        golden = golden_saliency(annotation, instance_id, dataset)
        ap_instance = defaultdict(float)
        for attribution_method in __attr_methods__:
            for aggr, attr in df.loc[instance_id, attribution_method].items():
                attr = scale_to_unit_interval(attr)
                attr = [max([attr[t] for t in w]) for w in instance['word_structure']['student_answer']]
                ap = average_precision_score(golden, attr)
                ap_instance[attribution_method + '_' + aggr] = ap
                ap_scores.append(ap_instance)
    ha = pd.DataFrame.from_records(ap_scores)
    ha = ha.mean(axis=0).to_dict()
    return ha

def activation_diff_models(model1, model2, batch, token_types):
    if token_types:
        act1 = get_layer_activations(model1, input_ids = batch.input, token_type_ids = batch.token_type_ids, attention_mask = batch.generate_mask())
        act2 = get_layer_activations(model2, input_ids = batch.input, token_type_ids = batch.token_type_ids, attention_mask = batch.generate_mask())
    else:
        act1 = get_layer_activations(model1, input_ids = batch.input, attention_mask = batch.generate_mask())
        act2 = get_layer_activations(model2, input_ids = batch.input, attention_mask = batch.generate_mask())
    assert act1.keys() == act2.keys()
    diff = np.mean([(act1[key] - act2[key]).abs().mean().cpu().item() for key in act1.keys()])
    return diff

def activation_diff_batches(model, pair, token_types):
    input_ids1,input_ids2 = torch.split(pair.input,1)
    attention_mask1, attention_mask2 = torch.split(pair.generate_mask(), 1)
    if token_types:
        token_types1,token_types2 = torch.split(pair.token_type_ids,1)
        act1 = get_layer_activations(model, input_ids = input_ids1, token_type_ids = token_type_ids1, attention_mask = attention_mask1)
        act2 = get_layer_activations(model, input_ids = input_ids2, token_type_ids = batch2.token_type_ids, attention_mask = attention_mask2)
    else:
        act1 = get_layer_activations(model,  input_ids = input_ids1, attention_mask = attention_mask1)
        act2 = get_layer_activations(model,  input_ids = input_ids2, attention_mask = attention_mask2)
    assert  act1.keys() == act2.keys()
    diff = np.mean([(act1[key] - act2[key]).abs().mean().cpu().item() for key in act1.keys()])
    return diff

def attribution_diff(attr1, attr2):
    attr1 = scale_to_unit_interval(attr1)
    attr2 = scale_to_unit_interval(attr2)
    return np.mean(np.abs(np.array(attr1) - np.array(attr2)))

# assumes attr are lists
def pad_attributions(attr1, attr2):
    if len(attr1) < len(attr2):
        attr1 += [0.0]*(len(attr2)-len(attr1))
    if len(attr2) < len(attr1):
        attr2 += [0.0]*(len(attr1)-len(attr2))
    return attr1, attr2

def compute_rationale_consistency(attr_data1, attr_data2, cuda = False):
    if not attr_data1.is_compatible(attr_data2):
        raise Exception('Can only compute rationale consistency for compatible AttributionData.')
    attr_aggr_list = [attribution_method + '_' + aggr for attribution_method in __attr_methods__ for aggr in __aggr__]
    try:
        model1, config1 = attr_data1.load_model()
        model2, config2 = attr_data2.load_model()
    except:
        return {attr_aggr: np.nan for attr_aggr in attr_aggr_list}
    model1.eval()
    model2.eval()
    if cuda:
        model1.cuda()
        model2.cuda()
    attr_data1.set_attr_class('pred')
    attr_data2.set_attr_class('pred')
    df1 = attr_data1.df.set_index('instance_id')
    df2 = attr_data2.df.set_index('instance_id')
    diffs = []
    token_types = attr_data1.token_types
    loader = attr_data1.get_dataloader()
    with tqdm(total=len(loader.batch_sampler)) as pbar:
        pbar.set_description(f'Computing rationale consitencys:  {attr_data1.run_id}, {attr_data2.run_id}')
        for batch in loader:
            diff_instance = defaultdict()
            instance_id = batch.instance.item()
            if cuda:
                batch.cuda()
            act_diff = activation_diff_models(model1, model2, batch, token_types)
            # act_diff =  np.random.rand()
            diff_instance['Activation'] = act_diff
            for attribution_method in __attr_methods__:
                for aggr in __aggr__:
                    attr1 = df1.loc[instance_id, attribution_method][aggr]
                    attr2 = df2.loc[instance_id, attribution_method][aggr]
                    attr_diff = attribution_diff(attr1, attr2)
                    # attr_diff =  np.random.rand()
                    diff_instance[attribution_method + '_' + aggr] = attr_diff
            diffs.append(diff_instance)
            batch.cpu()
            pbar.update(1)
    model1.cpu()
    model2.cpu()
    df_diffs = pd.DataFrame.from_records(diffs)
    r_scores = {col: spearmanr(df_diffs[['Activation', col]])[0] for col in df_diffs.columns if not 'Activation' in col}
    return r_scores


def compute_dataset_consistency(attr_data, cuda = False, **kwargs):
    token_types = attr_data.token_types
    model, config = attr_data.load_model()
    model.eval()
    if cuda:
        model.cuda()
    attr_data.set_attr_class('pred')
    df = attr_data.df.set_index('instance_id')
    diffs = []
    loader = attr_data.get_pairdataloader(**kwargs)
    with tqdm(total=len(loader.batch_sampler))  as pbar:
        pbar.set_description(f'Computing dataset consitency:  {attr_data.run_id}.')
        for pair in loader:
            instance_id1, instance_id2 = pair.instance.squeeze().numpy().tolist()
            if cuda:
                pair.cuda()
            diff_instance = defaultdict()
            act_diff = activation_diff_batches(model, pair, token_types)
            #act_diff = np.random.rand()
            diff_instance['Activation'] = act_diff
            for attribution_method in __attr_methods__:
                attributions1 = df.loc[instance_id1, attribution_method]
                attributions2 = df.loc[instance_id2, attribution_method]
                for aggr in __aggr__:
                    attr1 = attributions1[aggr]
                    attr2 = attributions2[aggr]
                    attr1, attr2 = pad_attributions(attr1, attr2)
                    attr_diff = attribution_diff(attr1, attr2)
                    #attr_diff =  np.random.rand()
                    diff_instance[attribution_method + '_' + aggr] = attr_diff
            diffs.append(diff_instance)
            pair.cpu()
            pbar.update(1)
    model.cpu()
    df_diffs = pd.DataFrame.from_records(diffs)
    r_scores = {col: spearmanr(df_diffs[['Activation', col]])[0] for col in df_diffs.columns if not 'Activation' in col}
    return r_scores
