import os
import pandas as pd
import numpy as np
import configuration
import transformers
import torch
from sklearn.metrics import average_precision_score

DATA_FILE = 'data/flat_semeval5way_test.csv'

def get_annotations(annotation_dir):
    df = pd.DataFrame(columns = ['name','source','origin','annotation', 'question_id'])
    for ann in os.scandir(annotation_dir):
        name, source, origin1, origin2, question_id = ann.name.split('.')[0].split('_')
        origin = '_'.join([origin1,origin2])
        with open(os.path.join(annotation_dir, ann.name), 'r') as f:
            annotation = f.readline()[:-1]
        df = df.append({'name': name, 'source': source, 'origin':origin, 'annotation':annotation, 'question_id':question_id}, ignore_index = True)
    return df

def list_word_token_idx(text, tokenizer, special_tokens = False):
    word_encodings =  [tokenizer.encode(word, add_special_tokens = special_tokens) for word in text.split()]
    it = iter(range(sum(len(w) for w in word_encodings)))
    word_idx = [[next(it) for j in w] for w in word_encodings]
    return word_idx

def compute_golden_saliency_vector(annotation_row, data_row):
    return np.array([1 if f'word_{k+1}' in annotation_row['annotation'].split(',') else 0 for k in range(len( data_row['student_answers'].split()))])

def compute_student_answer_word_saliency_vector(attribution_row, data_row, tokenizer, aggr):
    # First we select the attribibtuion corresponding to the student_answer_tokens (without [SEP])
    question_tokens = tokenizer.encode(data_row['question_text'])
    reference_tokens = tokenizer.encode(data_row['reference_answers'])
    student_tokens = tokenizer.encode(data_row['student_answers'])
    token_type_ids = [0] + [1]*(len(question_tokens)-1) + [2]*(len(reference_tokens)-1) + [3]*(len(student_tokens)-1)
    attr = attribution_row['attr_' + aggr][[id ==3 for id in token_type_ids]][:-1]
    # Then we map this to word attributions and normalize
    # NOTE: choices below may be changed
    w_idx = list_word_token_idx(data_row['student_answers'], tokenizer)
    assert(len(attr) == sum(len(w) for w in w_idx))
    attr = [np.max(attr[w]) for w in w_idx]
    attr = np.array(attr)/np.max(attr)
    return attr

def compute_human_agreement(attribution_file, annotation_dir, aggr = 'norm'):
    name, source = os.path.split(attribution_file)[-1].split('_')[0:2]
    print('Compute human agreement (HA) for', name, source)
    model_name = configuration.load(name).model_name
    tokenizer =  transformers.AutoTokenizer.from_pretrained(model_name, lowercase=True)
    df_anno = get_annotations(annotation_dir)
    df_anno = df_anno[df_anno.source == source]
    df_attr = pd.read_pickle(attribution_file)
    df_attr = df_attr[df_attr['pred']==df_attr['attr_class']]
    df = pd.read_csv(DATA_FILE)
    df['question_id'] = df.index
    df = df[(df.source == source)&(df.origin.str.contains('answer'))]
    assert(len(df) == len(df_attr))
    df_attr.index = df['question_id']
    df.index = df['question_id']
    AP = [None]*len(df_anno)
    for i in range(len(df_anno)):
        annotation_row = df_anno.iloc[i]
        data_row = df.loc[int(annotation_row['question_id'])]
        attribution_row = df_attr.loc[int(annotation_row['question_id'])]
        golden = compute_golden_saliency_vector(annotation_row, data_row)
        saliency = compute_student_answer_word_saliency_vector(attribution_row, data_row, tokenizer, aggr)
        ap = average_precision_score(golden, saliency)
        AP[i] = ap
        #print('AP is ', ap)

    df_anno['AP'] = AP
    MAP = df_anno['AP'].mean()
    print('MAP is ', ap)
    return MAP
