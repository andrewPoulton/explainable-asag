import pandas as pd
from wandbinteraction import get_runs, get_run_info
import os
import numpy as np
from evaluation import __attr_aggr__, __attr_methods__, __aggr__, RCmetric
from evaluation import RCmetric
from dataset import dataloader,__TEST_DATA__

models = [
    'bert-base',
    'bert-large',
    'roberta-base',
    'roberta-base',
    'roberta-base',
    'roberta-large',
    'albert-base',
    'albert-large',
    'albert-large',
    'distilbert-base',
    'distilroberta',
    'distilbert-base-squad2',
    'roberta-base-squad2',
    'distilroberta-base-squad2',
    'bert-base-squad2',
    'albert-base-squad2']

def download_experiment_info(outfile = False):
    my_stats_list = []
    summary_list = []
    config_list = []
    for run in get_runs():
        # add some stats I want
        max_f1 = 0.0
        for i, row in run.history(keys = ['f1']).iterrows():
            if row['f1'] > max_f1:
                max_f1 = row['f1']
        max_f1_macro = 0.0
        for i, row in run.history(keys = ['f1-macro']).iterrows():
            if row['f1-macro'] > max_f1_macro:
                max_f1_macro = row['f1-macro']

        my_stats_list.append({'run_id':run.id, 'state':run.state, 'max_f1': max_f1, 'max_f1_macro':max_f1_macro})

        # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # run.config is the input metrics.  We remove special values that start with _.
        config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')})

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    my_stats_df = pd.DataFrame.from_records(my_stats_list)
    df = pd.concat([my_stats_df, config_df, summary_df], axis=1)
    if outfile:
        df.to_csv(outfile)
    return df

def experiment_info_to_trainingresults(in_file_or_df, outfile = False):
    if isinstance(in_file_or_df, pd.DataFrame):
        df = in_file_or_df
    elif isinstance(in_file_or_df, str) and in_file_or_df.endswith('.csv'):
        df = pd.read_csv(in_file_or_df)
    else:
        Exception('Need dataframe or csv file.')
    dfs = []
    groups = ['scientsbank', 'scientsbank-token_types', 'beetle', 'beetle-token_types']
    group_cols = ['token_types', 'data_source', 'name', 'model_name']
    run_cols = ['run_id', 'state', 'max_f1', 'max_f1_macro']
    cols = run_cols + group_cols
    for group in groups:
        df_g1 = df[df['group'] == group + '-1']
        df_g2 = df[df['group'] == group + '-2']
        df_g3 = df[df['group'] == group + '-3']
        dfs_group = []
        for i, df_g in enumerate([df_g1, df_g2, df_g2]):
            df_g = df_g[cols]
            df_g = df_g.set_index(group_cols)
            df_g.columns = [col + '-' + str(i+1) for col in df_g.columns]
            dfs_group.append(df_g)
        df_group = pd.concat(dfs_group, axis = 1)
        df_group = df_group.reset_index()
        dfs.append(df_group)
    df = pd.concat(dfs, axis = 0)
    df.reset_index(inplace = True, drop = True)
    df.rename(inplace = True, columns = {'index':'run_index',
                                         'data_source': 'source',
                                         'name':'model',
                                         'model_name':'model_path'})
    df['model_index'] = df['model'].apply(lambda m: models.index(m))
    if outfile:
        df.to_csv(outfile)
    return df

def join_evaluation_results(evaluations_dir=os.path.join('results', 'evaluations')):
    groups = ['scientsbank', 'scientsbank-token_types', 'beetle', 'beetle-token_types']
    metrics = ['RC', 'DC', 'HA']
    dfs = []
    for group in groups:
        for metric in metrics:
            filepath = os.path.join(evaluations_dir, group + '_' + metric + '.csv')
            try:
                _df = pd.read_csv(filepath, index_col = 0)
                _df = _df.rename(columns = {'model_name': 'model'})
                _df['metric'] = metric
                if metric == 'HA':
                    _info = pd.DataFrame.from_records([get_run_info(run) for run in _df['run_id']])
                    _df = pd.concat([_df, _info], axis = 1)
                dfs.append(_df)
                #_df.drop(columns = ['attr_file'], inplace = True)
            except IOError:
                print('Something went wrong, probably no file:', filepath)
                pass

    df = pd.concat(dfs, axis = 0)
    df.reset_index(drop = True, inplace = True)
    df.drop(['run_id', 'run_id1', 'run_id2', 'attr_file'], axis = 1, inplace = True)
    df['model_index'] = df['model'].apply(lambda m: models.index(m) if isinstance(m, str) else np.nan).astype(int)
    df.to_csv(os.path.join('results','evaluationresults.csv'))
    return df


def drop_not_L2(df):
    columns = [col for col in df.columns if not (col.endswith('_L1') or col.endswith('_sum'))]
    df = df[columns]
    df.columns = [col.replace('_L2','') if col.endswith('_L2') else col for col in df.columns]
    return df

def RC_from_raw(in_dir, metric = RCmetric):
    ds = []
    for f in os.listdir(in_dir):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(in_dir, f), index_col = 0)
            d = {col: metric(df['Activation'], df[col]) for col in __attr_aggr__ + ['Random']}
            d.update({
                'model_name': df['model_name'][0],
                'source': df['source'][0],
                'token_types': bool(df['token_types'][0])
            })
            ds.append(d)
        else:
            continue
    df = pd.DataFrame.from_records(ds)
    df = drop_not_L2(df)
    return df

def save_RC():
    RCdirs = [
    'raw/RC/beetle',
    'raw/RC/scientsbank',
    'raw/RC/beetle-token_types',
    'raw/RC/scientsbank-token_types']
    dfs = []
    for d in RCdirs:
        dfs.append(RC_from_raw(d))

    df = pd.concat(dfs, axis = 0)
    df.to_csv('results/RC.csv')


def save_HA():
    HAdir = 'raw/HA'
    dfs = []
    for f in os.listdir(HAdir):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(HAdir, f), index_col = 0)
            info = df['run_id'].apply(get_run_info).tolist()
            info =  pd.DataFrame.from_records(info)
            df = df.drop(columns =['run_id', 'attr_file'])
            df = drop_not_L2(df)
            df = pd.concat([df, info], axis = 1)
            df = df.rename(columns = {'model':'model_name'})
            dfs.append(df)
        else:
            continue
    df = pd.concat(dfs, axis = 0)
    df.to_csv('results/HA.csv')
