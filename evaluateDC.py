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

def evaluate_dataset_consistency(*attribution_files, backupfile = None, **kwargs):
    try:
        dc_list = load_pickle(backupfile)
        run_ids = [it['run_id'] for it in dc_list]
    except:
        dc_list = []
        run_ids = []
    for attr_file in attribution_files:
        if any(run_id in attr_file for run_id in run_ids):
            continue
        else:
            attr_data = AttributionData(attr_file)
            dc = compute_dataset_consistency(attr_data, **kwargs)
            dc.update({'run_id': attr_data.run_id,
                       'source': attr_data.source,
                       'token_types': attr_data.token_types})
            dc_list.append(dc)
            if backupfile:
                to_pickle(dc_list, backupfile)
    return pd.DataFrame.from_records(dc_list)

def evaluateDC(attributions_dir, selection = True):
    if not os.path.isdir( __RESULTS_DIR__):
        os.mkdir(__RESULTS_DIR__)

    attr_files = [os.path.join(attributions_dir, f) for f in os.listdir(attributions_dir) if f.endswith('.pkl')]
    path_pieces =  os.path.normpath(attributions_dir).split(os.sep)

    group = re.sub('\-[0-9]$', '', path_pieces[-1])
    if selection:
        filepath = os.path.join(__RESULTS_DIR__, group + '_DC.csv')
        backupfile = os.path.join(__RESULTS_DIR__, group,  'DC.pkl')
        kwargs = {'within_questions': True,
                  'between_questions': True}
    else:
        filepath = os.path.join(__RESULTS_DIR__, group + '_DC_all.csv')
        backupfile = os.path.join(__RESULTS_DIR__, group,  'DC_all.pkl')
        kwargs = {}
    if not os.path.isdir(os.path.join(__RESULTS_DIR__,group)):
        os.mkdir(os.path.join(__RESULTS_DIR__,group))
    to_pickle([], backupfile)
    pd.DataFrame().to_csv(filepath)
    df = evaluate_dataset_consistency(*attr_files, backupfile = backupfile  , cuda = __CUDA__,**kwargs)
    df.to_csv(filepath)

if __name__ == '__main__':
    fire.Fire(evaluateDC)
