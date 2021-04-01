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
from filehandling import to_json, load_json
from sklearn.metrics.pairwise import cosine_similarity
#warnings.filterwarnings("error")
#['IntegratedGradients', 'InputXGradient','Saliency','GradientShap','Occlusion']
__RESULTS_DIR__ = 'evaluations'
__attr_methods__ = ['GradientShap', 'InputXGradient', 'IntegratedGradients', 'Occlusion','Saliency']
#__aggr__ = ['L2', 'L1', 'sum']
__aggr__ = ['L2']
__attr_aggr__ = [m + '_' + a if m!='Random' else m for m in __attr_methods__ for a in __aggr__]

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
    def __init__(self, attribution_file_or_df):
        if isinstance(attribution_file_or_df, pd.DataFrame):
            self.df = df
        if isinstance(attribution_file_or_df, str) and attribution_file_or_df.endswith('.json'):
            self.df = AttributionData.from_json(attribution_file_or_df)
        else:
            self.df  = pd.read_pickle(attribution_file_or_df)
        self.run_id = self.df['run_id'].unique()[0]
        self.model_name = self.df['model'].unique()[0]
        self.model_path = self.df['model_path'].unique()[0]
        self.source = self.df['source'].unique()[0]
        self.num_labels = self.df['num_labels'].unique()[0]
        self.origin = self.df['origin'].unique()[0]
        self.group = self.df['group'].unique()[0]
        self.token_types = self.df.iloc[0].get('token_types', False)
        self.df['token_types'] = self.token_types
        self.info_columns = ['run_id', 'model', 'model_path', 'source', 'origin', 'num_label', 'group', 'token_types', 'attributions']
        self.attr_methods = __attr_methods__
        self.attr_class = None
        self._df = self.df.copy()

    def to_json(self, filepath):
        info = {
            'run_id': self.run_id,
            'model': self.model_name,
            'model_path': self.model_path,
            'source': self.source,
            'origin': self.origin,
            'num_labels': self.num_labels,
            'group': self.group,
            'token_types': self.token_types,
        }
        info = stringify(info)
        explained = self.df[self.df.columns.difference(self.info_columns)]
        explained = explained.to_dict(orient = 'records')
        explained = stringify(explained)
        to_json({'info':info, 'explained':explained},filepath)

    @staticmethod
    def from_json(filepath):
        attr_data = load_json(filepath)
        info = attr_data['info']
        explained = attr_data['explained']
        info = {'run_id': info['run_id'],
            'model': info['model'],
            'model_path': info['model_path'],
            'source': info['source'],
            'origin': info['origin'],
            'num_labels': int(info['num_labels']),
            'group': info['group'],
            'token_types': bool(info['token_types'])
        }
        df = pd.DataFrame.from_records(explained)
        df = df.astype({
            'instance_id':int,
            'label':int,
            'pred': int,
            'attr_class':int,
            'attr_class_pred_prob':float,
            'num_labels':int
                       })
        for attr_meth in __attr_methods__:
            df[attr_meth] = df[attr_meth].apply(lambda d: {k: [float(a) for a in l] for k,l in d.items()})
        for k,v in info.items():
            df[k] = v
        return AttributionData(df)

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
            num_workers = 2 if  torch.cuda.is_available() else 0)
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
        if self.run_id == 'random':
            model = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels = self.num_labels)
            model.init_weights()
            config = model.config.__dict__
        else:
            model, config = load_model_from_run(self.run_id)
        return model, config


def stringify(o):
    if isinstance(o, dict):
        return {stringify(k):stringify(v) for k,v in o.items()}
    elif isinstance(o, list):
        return [stringify(i) for i in o]
    else:
        return str(o)

def golden_saliency(annotation, instance_id, dataset):
    student_answer_words = dataset.get_instance(instance_id)['student_answers'].split()
    golden =  [1 if f'word_{k}' in annotation.split(',')
               else 0 for k,w in enumerate(student_answer_words)]
    return golden

def ann_aggr_function(anns):
    return ','.join([word_k for word_k, i in Counter(chain(anns)).items() if i > 1])

def scale_to_unit_interval(attr, aggr):
    _attr = MinMaxScaler().fit_transform([[np.abs(a)] for a in attr])
    if aggr == 'sum':
        _attr =  [np.sign(a)*b for a, b in zip(attr, _attr)]
    return [a[0] for a in _attr]

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

def cos_sim(attr1, attr2):
    return cosine_similarity([attr1],[attr2]).item()

def RCmetric(diff_act, diff_attr):
    D = torch.Tensor(diff_act)
    S = 1.0 - torch.Tensor(diff_attr)
    return torch.sum(S*torch.nn.functional.softmax(D/torch.mean(D), dim = 0)).item()

# def attribution_diff(attr1, attr2):
#     ap1 = average_precision_score([round(a) for a in attr1], attr2)
#     ap2 = average_precision_score([round(a) for a in attr2], attr1)
#     return 0.5*(ap1+ap2)

def attribution_diff(attr1, attr2):
    return 1.0 - cos_sim(attr1, attr2)

def compute_human_agreement(attr_data, ann_data, return_df = False):
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
                attr = scale_to_unit_interval(attr, aggr)
                if aggr == 'sum':
                    attr = [abs(a) for a in attr]
                attr = [max([attr[t] for t in w]) for w in instance['word_structure']['student_answer']]
                ap = average_precision_score(golden, attr)
                ap_instance[attribution_method + '_' + aggr] = ap
                ap_scores.append(ap_instance)
    ap_df = pd.DataFrame.from_records(ap_scores)
    ha = ap_df.mean(axis=0).to_dict()
    if return_df:
        return ha, ap_df
    else:
        return ha

def compute_rationale_consistency(attr_data1, attr_data2, cuda = False, return_df = True, scale = True):
    if not attr_data1.is_compatible(attr_data2):
        raise Exception('Can only compute rationale consistency for compatible AttributionData.')
    attr_aggr_list = [attribution_method + '_' + aggr for attribution_method in __attr_methods__ for aggr in __aggr__]
    model1, config1 = attr_data1.load_model()
    model2, config2 = attr_data2.load_model()
    model1.eval()
    model2.eval()
    if cuda:
        model1.cuda()
        model2.cuda()
    # attr_data1.set_attr_class('pred')
    # attr_data2.set_attr_class('pred')
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
            df1 = attr_data1.df[attr_data1.df['instance_id']==instance_id]
            df2 = attr_data2.df[attr_data1.df['instance_id']==instance_id]
            assert (df1['attr_class'] == df2['attr_class']).all()
            attr_classes = df1['attr_class'].tolist()
            df1 = df1.set_index('attr_class', drop = True)
            df2 = df2.set_index('attr_class', drop = True)
            for attr_class in attr_classes:
                diff_instance['Activation'] = act_diff
                #for attribution_method in __attr_methods__:
                for attribution_method in ['GradientShap', 'InputXGradient', 'IntegratedGradients', 'Occlusion','Saliency', 'Random']:
                    if attribution_method == 'Random':
                        n = batch.input.size(1)
                        attr1 = [np.random.rand() for _ in range(n)]
                        attr2 = [np.random.rand() for _ in range(n)]
                        if scale:
                            attr1 = scale_to_unit_interval(attr1, 'L2')
                            attr2 = scale_to_unit_interval(attr2, 'L2')
                        diff_instance['Random'] = attribution_diff(attr1, attr2)
                    else:
                        for aggr in ['L2']:
                        #for aggr in __aggr__:
                            attr1 = df1.loc[attr_class, attribution_method][aggr]
                            attr2 = df2.loc[attr_class, attribution_method][aggr]
                            if scale:
                                attr1 = scale_to_unit_interval(attr1, aggr)
                                attr2 = scale_to_unit_interval(attr2, aggr)
                            # if overlap:
                            #     attr_diff = attribution_diff_overlap(attr1, attr2)
                            # else:
                            #     attr_diff = attribution_diff(attr1, attr2)
                            # attr_diff =  np.random.rand()
                            diff_instance[attribution_method + '_' + aggr] = attribution_diff(attr1, attr2)
            diff_instance['attr_class'] = attr_class
            diff_instance['pred_1'] = df1.loc[attr_class, 'pred']
            diff_instance['pred_2'] = df2.loc[attr_class, 'pred']
            diffs.append(diff_instance)
            batch.cpu()
            pbar.update(1)
    model1.cpu()
    model2.cpu()
    df_diffs = pd.DataFrame.from_records(diffs)
    scores = {col: RCmetric(df_diffs['Activation'], df_diffs[col]) for col in ['GradientShap_L2', 'InputXGradient_L2', 'IntegratedGradients_L2', 'Occlusion_L2','Saliency_L2', 'Random']}
    if return_df:
        return scores, df_diffs
    else:
        return r_scores


 # def compute_dataset_consistency(attr_data, cuda = False, return_df = False, **kwargs):
 #    token_types = attr_data.token_types
 #    model, config = attr_data.load_model()
 #    model.eval()
 #    if cuda:
 #        model.cuda()
 #    attr_data.set_attr_class('pred')
 #    df = attr_data.df.set_index('instance_id')
 #    diffs = []
 #    loader = attr_data.get_pairdataloader(**kwargs)
 #    with tqdm(total=len(loader.batch_sampler))  as pbar:
 #        pbar.set_description(f'Computing dataset consitency:  {attr_data.run_id}.')
 #        for pair in loader:
 #            instance_id1, instance_id2 = pair.instance.squeeze().numpy().tolist()
 #            if cuda:
 #                pair.cuda()
 #            diff_instance = defaultdict()
 #            act_diff = activation_diff_batches(model, pair, token_types)
 #            #act_diff = np.random.rand()
 #            diff_instance['Activation'] = act_diff
 #            for attribution_method in __attr_methods__:
 #                attributions1 = df.loc[instance_id1, attribution_method]
 #                attributions2 = df.loc[instance_id2, attribution_method]
 #                for aggr in __aggr__:
 #                    attr1 = attributions1[aggr]
 #                    attr2 = attributions2[aggr]
 #                    attr1, attr2 = pad_attributions(attr1, attr2)
 #                    # attr1 = scale_to_unit_interval(attr1, aggr)
 #                    # attr2 = scale_to_unit_interval(attr2, aggr)
 #                    attr_diff = attribution_diff(attr1, attr2)
 #                    #attr_diff =  np.random.rand()
 #                    diff_instance[attribution_method + '_' + aggr] = attr_diff
 #            diffs.append(diff_instance)
 #            pair.cpu()
 #            pbar.update(1)
 #    model.cpu()
 #    df_diffs = pd.DataFrame.from_records(diffs)
 #    r_scores = {col: spearmanr(df_diffs[['Activation', col]])[0] for col in df_diffs.columns if not 'Activation' in col}
 #    if return_df:
 #        return r_scores, df_diffs
 #    else:
 #        return r_scores



# # assumes attr are lists
# def pad_attributions(attr1, attr2):
#     if len(attr1) < len(attr2):
#         attr1 += [0.0]*(len(attr2)-len(attr1))
#     if len(attr2) < len(attr1):
#         attr2 += [0.0]*(len(attr1)-len(attr2))
#     return attr1, attr2



# def activation_diff_batches(model, pair, token_types):
#     input_ids1,input_ids2 = torch.split(pair.input,1)
#     attention_mask1, attention_mask2 = torch.split(pair.generate_mask(), 1)
#     if token_types:
#         token_type_ids1,token_type_ids2 = torch.split(pair.token_type_ids,1)
#         act1 = get_layer_activations(model, input_ids = input_ids1, token_type_ids = token_type_ids1, attention_mask = attention_mask1)
#         act2 = get_layer_activations(model, input_ids = input_ids2, token_type_ids = token_type_ids2, attention_mask = attention_mask2)
#     else:
#         act1 = get_layer_activations(model,  input_ids = input_ids1, attention_mask = attention_mask1)
#         act2 = get_layer_activations(model,  input_ids = input_ids2, attention_mask = attention_mask2)
#     assert  act1.keys() == act2.keys()
#     diff = np.mean([(act1[key] - act2[key]).abs().mean().cpu().item() for key in act1.keys()])
#     return diff

# def attribution_diff(attr1, attr2):
#     return np.mean(np.abs(np.array(attr1) - np.array(attr2)))

# def attribution_diff_overlap(attr1, attr2):
#     return 1 - np.dot(attr1,attr2)/np.linalg.norm(attr1)/np.linalg.norm(attr2)
