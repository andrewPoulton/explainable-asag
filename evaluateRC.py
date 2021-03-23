import fire
import os
import pandas as pd
from evaluation import (
    AnnotationData,
    AttributionData,
    compute_human_agreement,
    compute_rationale_consistency,
    compute_dataset_consistency
)
import torch
from wandbinteraction import get_runs

__CUDA__ = torch.cuda.is_available()
__ATTRIBUTION_DIR__ = 'attributions'
__RESULTS_DIR__ = 'results'


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
    if attribution_dir.endswith('/'):
        attribution_dir = attribution_dir[:-1]
    runs1 = get_runs(group1)
    runs2 = get_runs(group2)
    pairs = {run.config['name']: [os.path.join(attribution_dir, group1, run.id +'.pkl')] for run in runs1 }
    # pairs = {run.config['name']:  [run.config['model_name']] for run in runs1 }
    for run in runs2:
        pairs[run.config['name']].append(os.path.join(attribution_dir, group2, run.id +'.pkl'))
    # for run in runs2:
    #     pairs[run.config['name']].append(run.config['model_name'])
    # print(*pairs.values())
    df =  evaluate_rationale_consistency(*pairs.values())
    df.to_csv(os.path.join(__RESULTS_DIR__, attribution_dir, group1 + group + '_RC.csv'))

if __name__ == '__main__':
    fire.Fire(evaluateRC)
