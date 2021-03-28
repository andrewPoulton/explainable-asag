import pandas as pd
from evaluation import __attr_aggr__, __attr_methods__

def trainingresults():
    df = pd.read_csv('results/trainingresults.csv', index_col = 0)
    dfs = []
    for source in 'scientsbank', 'beetle':
        for token_types in False, True:
            _df = df[(df['source'] == source)&(df['token_types'] == token_types)]
            _df = _df[['model', 'model_index', 'max_f1-1', 'max_f1-2', 'max_f1-3']]
            _df = _df.set_index(['model_index','model'])
            _df[source + ('-TT-' if token_types else '-') + 'F1-mean'] = _df[['max_f1-1','max_f1-2','max_f1-3']].mean(axis = 1)
            _df[source + ('-TT-' if token_types else '-') +'F1-std'] = _df[['max_f1-1','max_f1-2','max_f1-3']].std(axis = 1)
            _df = _df.drop(columns = ['max_f1-1','max_f1-2','max_f1-3'])
            dfs.append(_df)
    df = pd.concat(dfs, axis = 1)
    df = (100*df).round(1)
    #df = df.round(3)
    df = df.reset_index().set_index('model_index').sort_index().set_index('model', drop = True)
    df.index.name = None
    df.to_csv(os.path.join('tables','trainingresults.tex'))
    tex = df.copy()
    for group in 'scientsbank', 'scientsbank-TT', 'beetle', 'beetle-TT':
        tex[group +'-F1'] = [f"${m} ({s})$" if m!= 0 else "" for m,s in zip(tex[group +'-F1-mean'],tex[group +'-F1-std'])]
        tex = tex.drop(columns = [group + '-F1-mean', group + '-F1-std'])
    tex = tex.to_latex(column_format = 'l')
    with open(os.path.join('tables', 'trainingresults.tex'), 'w') as f:
        f.write(tex)
    return df, tex

def evaluationresults(aggr = 'L2', source = 'scientsbank'):
    df = pd.read_csv('results/evaluationresults.csv', index_col = 0)
    df = df[df['source']==source]
    df = df[['model_index','model', 'source', 'token_types', 'metric'] +
            [a for a in __attr_aggr__ if a.endswith(aggr)]]
    df = df.rename(columns = {a: a.replace('_' + aggr, '') for a in __attr_aggr__})
    dfs = []
    for metric in ['DC', 'RC', 'HA']:
        _df = df[df['metric'] == metric]
        _df = _df.set_index(['model_index','model', 'token_types'])
        _df = _df.drop(columns = ['metric', 'source'])
        _df.columns = [metric + '_' + col for col in _df.columns]
        if metric in ['DC', 'RC']:
            _df = 0.5*(_df + 1)
        _df = _df.round(2)
        dfs.append(_df)
    df = pd.concat(dfs, axis = 1)
    df.index = df.index.droplevel()
    df = df[[e + '_' + a for a in __attr_methods__ for e in ['DC', 'RC', 'HA'] ]]
    with open(os.path.join('tables', source + '_evaluations_' + aggr + '.tex'), 'w') as f:
        f.write(df.to_latex())
    return df


    # df = (df.set_index(['source','token_types','metric','model'])*10).round(2).reset_index()
    # df.to_csv(os.path.join('tables', 'evaluationresults_full.csv'))
    # aggregated table
    # df_mean = df.groupby(['source','token_types','metric']).mean()
    # df_std = df.groupby(['source','token_types','metric']).std()
    # df_std.columns = [col + '_std' for col in df_std.columns]
    # df_aggr = pd.concat([df_mean, df_std], axis =1)
    # df.to_csv(os.path.join('tables', 'evaluationresults.csv'))
    # df = df[['model','source','token_types', 'metric'] +
    #         [a in __attr_aggr__ if a.endswith('_L2')]]
    # return df


if __name__=='__main__':
    groups, df = f1_table()
    print(df)
    #save_table(df, 'trainingresults')
