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
from configuration import load_configs_from_file
import re
__CUDA__ = torch.cuda.is_available()


def evaluate_rationale_consistency(*pairs_of_attribution_files, backupfile = None):
    rc_list = []
    for attr_file1, attr_file2 in pairs_of_attribution_files:
        attr_data1 = AttributionData(attr_file1)
        attr_data2 = AttributionData(attr_file2)
        if attr_data1.is_compatible(attr_data2):
            rc = compute_rationale_consistency(attr_data1, attr_data2, __CUDA__)
            rc.update({'model_name': attr_data1.model_name,'run_id1':attr_data1.run_id, 'run_id2':attr_data2.run_id})
            rc_list.append(rc)
        if backupfile:
            to_pickle(rc_list, backupfile)
    return pd.DataFrame.from_records(rc_list)

def evaluateRC(attribution_dir1, attribution_dir2):
    attribution_dir1 =  os.path.normpath(attribution_dir1)
    attribution_dir2 =  os.path.normpath(attribution_dir2)
    group1 = attribution_dir1.split(os.sep)[-1]
    group2 = attribution_dir2.split(os.sep)[-1]
    group = re.sub('\-[0-9]$', '', group1)
    assert  group == re.sub('\-[0-9]$', '', group2), 'Need to evaluate RC using similar groups'
    runs1 = get_runs(group1)
    runs2 = get_runs(group2)
    model_names = load_configs_from_file(os.path.join('configs', 'evaluate.yml'))['models']
    run_pairs = {name: [] for name in model_names}
    for run in runs1:
        run_pairs[run.config['name']].append(run)
    for run in runs2:
        run_pairs[run.config['name']].append(run)
    run_pairs = [ [r1.id,r2.id] for (r1,r2) in run_pairs.values() if r1.state ==r2.state == 'finished']
    file_pairs = [ [os.path.join(attribution_dir1, r1 + '.pkl'), os.path.join(attribution_dir2, r2 + '.pkl')] for r1,r2 in run_pairs]
    print('EvaluateRC with pairs:',*file_pairs)
    filepath =  os.path.join(__RESULTS_DIR__, group + '_RC.pkl')
    backupfile  = os.path.join(__RESULTS_DIR__, group, 'RC.pkl')
    os.makedirs(os.path.join(__RESULTS_DIR__, group), exist_ok = True)
    to_pickle([], backupfile)
    pd.DataFrame().to_csv(filepath)
    df = evaluate_rationale_consistency(*file_pairs, backupfile = backupfile)
    df.to_csv(filepath)

if __name__ == '__main__':
    fire.Fire(evaluateRC)
