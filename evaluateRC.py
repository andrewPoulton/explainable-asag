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
from wandbinteraction import get_runs
from filehandling import load_pickle, to_pickle
import re
__CUDA__ = torch.cuda.is_available()


def evaluate_rationale_consistency(*pairs_of_attribution_files, backupfile = None):
    try:
        rc_list = load_pickle(backupfile)
        run_ids = [rc['run_id1'] for rec in rc_list]
    except:
        rc_list = []
        run_ids = []
    for attr_file1, attr_file2 in pairs_of_attribution_files:
        if any(run_id in attr_file1 for run_id in run_ids):
            continue
        attr_data1 = AttributionData(attr_file1)
        attr_data2 = AttributionData(attr_file2)
        if attr_data1.is_compatible(attr_data2):
            rc = compute_rationale_consistency(attr_data1, attr_data2, __CUDA__)
            rc.update({'model_name': attr_data1.model_name,'run_id1':attr_data1.run_id, 'run_id2':attr_data2.run_id})
            rc_list.append(rc)
        if backupfile:
            to_pickle(rc_list, backupfile)
    return pd.DataFrame.from_records(rc_list)

def evaluateRC(attribution_dir, group1, group2):
    if not os.path.isdir(__RESULTS_DIR__):
        os.mkdir(os.path.isdir(__RESULTS_DIR__))
    group = re.sub('\-[0-9]$', '', group1)
    assert  group == re.sub('\-[0-9]$', '', group1), 'Need to evaluate RC using similar groups'
    runs1 = get_runs(group1)
    runs2 = get_runs(group2)
    assert set(run.config['name'] for run in runs1) == set(run.config['name'] for run in runs2), 'Not compatible groups of runs.'
    run_pairs = {run.config['name']: [run] for run in runs1}
    for run in runs2:
        run_pairs[run.config['name']].append(run)
    run_pairs = [ (r1,r2) for (r1,r2) in run_pairs.values() if r1.state ==r2.state == 'finished']
    #print(run_pairs)
    get_attr_file = lambda g, run: os.path.join(attribution_dir, g, run.id +'.pkl')
    file_pairs = [[get_attr_file(group1, r1), get_attr_file(group2, r2)] for r1, r2 in run_pairs]
    print('File pairs:', *file_pairs)
    filepath = os.path.join(__RESULTS_DIR__, group  + '_RC.csv')
    backupfile  = os.path.join(__RESULTS_DIR__, group, 'RC.pkl')
    if not os.path.isdir(os.path.join(__RESULTS_DIR__,group)):
        os.mkdir(os.path.join(__RESULTS_DIR__,group))
    to_pickle([], backupfile)
    pd.DataFrame().to_csv(filepath)
    df = evaluate_rationale_consistency(*file_pairs, backupfile = backupfile)
    df.to_csv(filepath)

if __name__ == '__main__':
    fire.Fire(evaluateRC)
