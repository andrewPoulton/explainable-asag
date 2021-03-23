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

__CUDA__ = torch.cuda.is_available()


def evaluate_rationale_consistency(*pairs_of_attribution_files):
    rc_list = []
    for attr_file1, attr_file2 in pairs_of_attribution_files:
        attr_data1 = AttributionData(attr_file1)
        attr_data2 = AttributionData(attr_file2)
        if attr_data1.is_compatible(attr_data2):
            rc = compute_rationale_consistency(attr_data1, attr_data2, __CUDA__)
            rc.update({'run_id1':attr_data1.run_id, 'run_id2':attr_data2.run_id,
                       'attr_file1': attr_file1, 'attr_file2':attr_file2})
            rc_list.append(rc)
    return pd.DataFrame.from_records(rc_list)

def evaluateRC(attribution_dir, group1, group2):
    runs1 = get_runs(group1)
    runs2 = get_runs(group2)
    assert set(run.config['name'] for run in runs1) == set(run.config['name'] for run in runs2), 'Not compatible groups of runs.'
    run_pairs = {run.config['name']: run for run in runs1}
    for run in runs2:
        run_pairs[run.config['name']].append(run)
    get_attr_file = lambda group, run: os.path.join(attribution_dir, group, run.id +'.pkl')
    attr_file_pairs = [[get_attr_file(group1, run1), get_attr_file(group2,run2) for name, run1, run2 in run_pairs.items()]

    filename = group1 + group2 + '_RC.csv'

    df = evaluate_rationale_consistency(*attr_pairs.values())
    df.to_csv(os.path.join(__RESULTS_DIR__,filename))

if __name__ == '__main__':
    fire.Fire(evaluateRC)
