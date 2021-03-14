import os
import pandas as pd
import numpy as np
import configuration
import transformers
import torch
from sklearn.metrics import average_precision_score
from configuration import load_configs_from_file
from wandb_interaction import get_model_from_run_id
from dataset import dataloader
from scipy.stats import spearmanr

DATA_FILE = 'data/flat_semeval5way_test.csv'
ATTRIBUTION_DIR = 'explained'
ANNOTATION_DIR = '../annotator/annotations'


### Human Agreement

def get_tokenizer(model):
    model_name = load_configs_from_file(os.path.join('configs', 'pretrained.yml'))[model]['model_name']
    tokenizer =  transformers.AutoTokenizer.from_pretrained(model_name, lowercase=True)
    return tokenizer

def read_annotation(annotation_dir, annotation_file_name):
    annotator, source, origin1, origin2, question_id = annotation_file_name.split('.')[0].split('_')
    origin = '_'.join([origin1,origin2])
    with open(os.path.join(annotation_dir, annotation_file_name), 'r') as f:
        annotation = f.readline()[:-1]
    return {'annotator': annotator, 'source': source, 'origin': origin, 'question_id': question_id, 'annotation': annotation}

def get_annotations(annotation_dir):
    df = pd.DataFrame(columns = ['annotator','source','origin','annotation', 'question_id'])
    for annotation_file in os.scandir(annotation_dir):
        annotation_row = read_annotation(annotation_dir, annotation_file.name)
        df = df.append(annotation_row, ignore_index = True)
    return df

def read_attribution(attribution_dir, attribution_file_name, attr_is_pred = False):
    model, source, run_id, attribution_method = attribution_file_name.split('.')[0].split('_')
    df = pd.read_pickle(os.path.join(attribution_dir,attribution_file_name))
    if attr_is_pred:
        df = df[df['attr_class']== df['pred']]
    return {'df': df, 'model': model, 'source': source, 'run_id': run_id, 'attribution_method' : attribution_method}


def list_word_token_idx(text, tokenizer, special_tokens = False):
    """Returns  list of form [[0,1], [2], [3,4]] where sublist corresponds to words in text and indices to the positions of the word-piece tokens after encoding"""
    word_encodings =  [tokenizer.encode(word, add_special_tokens = special_tokens) for word in text.split()]
    it = iter(range(sum(len(w) for w in word_encodings)))
    word_idx = [[next(it) for j in w] for w in word_encodings]
    return word_idx

def compute_golden_saliency_vector(annotation, sentence):
    return np.array([1 if f'word_{k}' in annotation.split(',') else 0 for k in range(len( sentence.split()))])

def scale_to_unit_interval(attr):
    attr = np.array(attr)
    attr = (attr - attr.min())/(attr.max() - attr.min())
    return attr

def compute_student_answer_word_saliency_vector(attr, data_row, tokenizer):
    # We need to know the indices for student answer tokens and how they group as words
    question_tokens = tokenizer.encode(data_row['question_text'])
    reference_tokens = tokenizer.encode(data_row['reference_answers'])
    student_tokens = tokenizer.encode(data_row['student_answers'])
    start_student_tokens = len(question_tokens) + len(reference_tokens) - 2
    end_student_tokens = len(question_tokens) + len(reference_tokens) + len(student_tokens) - 4
    st_idx = range(start_student_tokens, end_student_tokens)
    w_idx = list_word_token_idx(data_row['student_answers'], tokenizer)
    assert(len(st_idx) == sum(len(w) for w in w_idx))
    # Then we normalize, select student answer (without [SEP]) and map to words
    attr = scale_to_unit_interval(attr)
    attr = attr[st_idx]
    attr = [np.max(attr[w]) for w in w_idx]
    return attr


def read_test_data(source):
    df = pd.read_csv(DATA_FILE)
    df['question_id'] = df.index
    df = df[(df.source == source)&(df.origin.str.contains('answer'))]
    df = df.set_index('question_id')
    return df

def compute_human_agreement(attribution_file_name, aggr = 'L2'):
    print('Computing human agreement (HA) for', attribution_file_name)
    attributions = read_attribution(ATTRIBUTION_DIR, attribution_file_name, attr_is_pred = True)
    tokenizer = get_tokenizer(attributions['model'])
    df_anno = get_annotations(ANNOTATION_DIR)
    df_anno = df_anno[df_anno.source == attributions['source']]
    df = read_test_data(attributions['source'])
    assert(len(df) == len(df_attr))
    df_attr.index = df.index
    AP = [None]*len(df_anno)
    for i in range(len(df_anno)):
        annotation_row = df_anno.iloc[i]
        data_row = df.loc[int(annotation_row['question_id'])]
        attribution_row = df_attr.loc[int(annotation_row['question_id'])]
        golden = compute_golden_saliency_vector(annotation_row['annotation'], data_row['student_answers'])
        saliency = compute_student_answer_word_saliency_vector(attribution_row['attr_' + aggr], data_row, tokenizer)
        ap = average_precision_score(golden, saliency)
        AP[i] = ap
        #print('AP is ', ap)

    df_anno['AP'] = AP
    MAP = df_anno['AP'].mean()
    return MAP



### Rationale Consistency
def compute_diff_activation(model1, model2, instance):
    return 0.0


def compute_diff_attribution(attr1, attr2):
    return np.mean(np.abs(np.array(attr1) -np.array(attr2)))


def compute_rationale_consistency(attribution_file1, attribution_file2, aggr = 'L2', **kwargs):
    A1 = read_attribution(ATTRIBUTION_DIR, attribution_file1, attr_is_pred=True)
    A2 = read_attribution(ATTRIBUTION_DIR, attribution_file2, attr_is_pred=True)
    assert A1['model'] == A2['model'] and A1['source'] == A2['source'] and A1['attribution_method'] == A2['attribution_method']
    model1, config1 = get_model_from_run_id(A1['run_id'], **kwargs)
    model2, config2 = get_model_from_run_id(A2['run_id'], **kwargs)
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
        diff_activation[i] = compute_diff_activation(model1, model2, batch.input)
        diff_attribution[i] = compute_diff_attribution(attr1, attr2)
    r = spearmanr(diff_activation, diff_attribution)
    return r
