import fire
import os
import pandas as pd
from evaluation import (
    AnnotationData,
    AttributionData,
    compute_human_agreement,
    compute_rationale_consistency,
    compute_dataset_consistency,
    __RESULTS_DIR__
)
import torch
import re
from filehandling import to_json, to_pickle

__CUDA__ = torch.cuda.is_available()

def evaluate_dataset_consistency(*attribution_files, datadir, **kwargs):
    dc_list = []
    for attr_file in attribution_files:
        attr_data = AttributionData(attr_file)
        dc = compute_dataset_consistency(attr_data, **kwargs)
        model_name = attr_data.model_name
        dc.update({'model': model_name,
                   'run_id':  attr_data.run_id,
                   'source': attr_data.source,
                   'token_types': attr_data.token_types})
        to_json({str(k): str(v) for k,v in dc.items()}, os.path.join(datadir, model_name + '.json'))
        dc_list.append(dc)
    return pd.DataFrame.from_records(dc_list)

def evaluateDC(attributions_dir, selection = True):
    attr_files = [os.path.join(attributions_dir, f) for f in os.listdir(attributions_dir) if f.endswith('.pkl')]
    print('EvaluateDC:', *attr_files)
    path_pieces =  os.path.normpath(attributions_dir).split(os.sep)
    group = re.sub('\-[0-9]$', '', path_pieces[-1])
    if selection:
        filepath = os.path.join(__RESULTS_DIR__, group + '_DC.csv')
        datadir = os.path.join(__RESULTS_DIR__, group,  'DC')
        kwargs = {'within_questions': True,
                  'between_questions': True}
    else:
        filepath = os.path.join(__RESULTS_DIR__, group + '_DC_all.csv')
        datadir = os.path.join(__RESULTS_DIR__, group,  'DC_all')
        kwargs = {}
    os.makedirs(datadir, exist_ok=True)
    df = evaluate_dataset_consistency(*attr_files, datadir = datadir  , cuda = __CUDA__,**kwargs)
    df.to_csv(filepath)

if __name__ == '__main__':
    fire.Fire(evaluateDC)
