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
from filehandling import load_pickle, to_pickle, load_json, to_json
from configuration import load_configs_from_file
import re
__CUDA__ = torch.cuda.is_available()


# def evaluate_rationale_consistency(datadir, *pairs_of_attribution_files):
#     #rc_list = []


#             # rc.update({'model_name': model_name,
#             #            'run_id1':attr_data1.run_id,
#             #            'run_id2':attr_data2.run_id,
#             #            'source': attr_data1.source,
#             #            'token_types': attr_data1.token_types})
#             # to_json({str(k):str(v) for k,v in  rc.items()}, os.path.join(datadir, model_name + '.json'))
#     #         rc_list.append(rc)
#     # return pd.DataFrame.from_records(rc_list)

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
    #filepath =  os.path.join(__RESULTS_DIR__, group + '_RC.csv')
    datadir  = os.path.join(__RESULTS_DIR__, group, 'RCscale')
    os.makedirs(datadir, exist_ok = True)
    for attr_file1, attr_file2 in file_pairs:
        attr_data1 = AttributionData(attr_file1)
        attr_data2 = AttributionData(attr_file2)
        model_name = attr_data1.model_name
        rc, df = compute_rationale_consistency(attr_data1, attr_data2, __CUDA__, return_df = True, scale = True)
        df['model_name'] = model_name
        df['run_id1'] = attr_data1.run_id
        df['run_id2'] = attr_data2.run_id
        df['source'] = attr_data1.source
        df['token_types'] = attr_data1.token_types
        df.to_csv(os.path.join(datadir, model_name + '.csv'))
    #df = evaluate_rationale_consistency(datadir, *file_pairs)
    # df.to_csv(filepath)

if __name__ == '__main__':
    fire.Fire(evaluateRC)
