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
from dataset import SemEvalDataset, dataloader
from tqdm import tqdm
from configuration import load_configs_from_file
import warnings
warnings.filterwarnings("error")


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

class AnnotationData:
    def __init__(self, annotation_dir):
        self.annotation_dir = annotation_dir
        self.load_annotations()
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
        self.run_id = self.df['run_id'].unique()[0]
        self.model = self.df['model'].unique()[0]
        self.model_path = self.df['model_path'].unique()[0]
        self.source = self.df['source'].unique()[0]
        self.num_labels = self.df['num_labels'].unique()[0]
        self.origin = self.df['origin'].unique()[0]
        self.group = self.df['group'].unique()[0]
        self.attr_methods = list(load_configs_from_file(os.path.join('configs','explain.yml'))['EXPLAIN'].keys())
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
            num_workers = 0)
        self.dataset = loader.dataset

    def compute_human_agreement(self, annotation_data):
        annotation_data.set_source(self.source)
        df = self.df[self.df['pred'] == self.df['attr_class']].set_index('instance_id')
        ap_scores = []
        for i, ann in tqdm(annotation_data.annotations.iterrows(), desc = 'Compute average precision scores'):
            annotation = ann['annotation']
            instance_id = ann['instance_id']
            instance = self.dataset.get_instance(instance_id, word_structure = True)
            golden = golden_saliency(annotation, instance_id, self.dataset)
            if not golden:
                df = df.drop(instance_id, axis=0)
            else:
                ap_instance = dict()
                for attribution_method in self.attr_methods:
                    for aggr, attr in df.loc[instance_id, attribution_method].items():
                        attr = scale_to_unit_interval(attr)
                        attr = [max([attr[t] for t in w]) for w in instance['word_structure']['student_answer']]
                        ap = average_precision_score(golden, attr)
                        ap_instance.update({attribution_method + '_' + aggr : ap})
                ap_scores.append(ap_instance)
        ha = pd.DataFrame.from_records(ap_scores)
        ha = ha.mean(axis=0).to_dict()
        return ha

def evaluate_human_agreement(annotations_dir, *attr_files):
    annotation_data = AnnotationData(annotations_dir)
    annotation_data.set_annotator('sebas')
    ha_list = []
    run_ids= []
    for attribution_file in attr_files:
        attribution_data = AttributionData(attribution_file)
        annotation_data.set_source(attribution_data.source)
        ha = attribution_data.compute_human_agreement(annotation_data)
        ha_list.append(ha)
        run_ids.append(attribution_data.run_id)

    ha_df = pd.DataFrame.from_records(ha_list)
    run_df = pd.DataFrame.from_dict({'run_id': run_ids})
    df = pd.concat([run_df, ha_df], axis = 1)
    return df


    # MAP = np.mean(AP)
    # return {'attribution_file_name': attribution_file_name,
    #         'metric': MAP,
    #         'run_id': attribution['run_id'],
    #         'model': attribution['model'],
    #         'source': attribution['source'],
    #         'attribution_method' : attribution['attribution_method'],
    #         'aggr': aggr}



### Rationale Consistency
###
### get the activations of layers is based on
### https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
### and
### https://github.com/copenlu/xai-benchmark/blob/master/saliency_eval/consist_data.py

# def is_layer(name):
#     layer_pattern = re.compile('^[a-z]*\.encoder\.layer.[0-9]*$')
#     return layer_pattern.search(name) or name == 'classifier'

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
