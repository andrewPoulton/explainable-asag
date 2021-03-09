import os
import pandas as pd
import numpy as np
import configuration
import transformers
import torch
from sklearn.metrics import average_precision_score
from configuration import load_configs_from_file

DATA_FILE = 'data/flat_semeval5way_test.csv'
ATTRIBUTION_DIR = 'explained'
ANNOTATION_DIR = '../annotator/annotations'

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

def read_attribution(attribution_dir, attribution_file_name):
    name, source, run_id, attribution_method = attribution_file_name.split('.')[0].split('_')
    df = pd.read_pickle(os.path.join(attribution_dir,attribution_file_name))
    return {'df': df, 'name': name, 'source': source, 'run_id': run_id, 'attribution_method' : attribution_method}


def list_word_token_idx(text, tokenizer, special_tokens = False):
    """Returns  list of form [[0,1], [2], [3,4]] where sublist corresponds to words in text and indices to the positions of the word-piece tokens after encoding"""
    word_encodings =  [tokenizer.encode(word, add_special_tokens = special_tokens) for word in text.split()]
    it = iter(range(sum(len(w) for w in word_encodings)))
    word_idx = [[next(it) for j in w] for w in word_encodings]
    return word_idx

def compute_golden_saliency_vector(annotation, data_row):
    return np.array([1 if f'word_{k+1}' in annotation.split(',') else 0 for k in range(len( data_row['student_answers'].split()))])

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

def compute_human_agreement(attribution_file_name, aggr = 'norm'):
    print('Computing human agreement (HA) for', attribution_file_name)
    attributions = read_attribution(ATTRIBUTION_DIR, attribution_file_name)
    model_name = load_configs_from_file(os.path.join('configs', 'pretrained.yml'))[attributions['name']]['model_name']
    tokenizer =  transformers.AutoTokenizer.from_pretrained(model_name, lowercase=True)
    df_attr = attributions['df'][attributions['df']['pred']==attributions['df']['attr_class']]
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
        golden = compute_golden_saliency_vector(annotation_row['annotation'], data_row)
        saliency = compute_student_answer_word_saliency_vector(attribution_row['attr_' + aggr], data_row, tokenizer)
        ap = average_precision_score(golden, saliency)
        AP[i] = ap
        #print('AP is ', ap)

    df_anno['AP'] = AP
    MAP = df_anno['AP'].mean()
    return MAP
