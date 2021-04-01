import pandas as pd
from evaluation import __attr_aggr__, __attr_methods__
import os
from functools import reduce

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
model_index = {m: models.index(m) for m in models}
to_model_index = lambda m: m.replace(model_index)

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
    #df = (100*df).round(1)
    #df = df.round(3)
    df = df.reset_index().set_index('model_index').sort_index().set_index('model', drop = True)
    df.index.name = None
    tex = df.copy()
    for group in 'scientsbank', 'scientsbank-TT', 'beetle', 'beetle-TT':
        #tex[group +'-F1'] = [f"${m} ({s})$" if m!= 0 else "" for m,s in zip(tex[group +'-F1-mean'],tex[group +'-F1-std'])]
        tex[group +'-F1'] = latex_col(100*tex[group +'-F1-mean'], std = 100*tex[group +'-F1-std'], digits = 1, std_digits = 2)
        tex = tex.drop(columns = [group + '-F1-mean', group + '-F1-std'])
    tex.to_csv(os.path.join('tables','trainingresults.csv'), sep = '&', index = True ,line_terminator = '\\\\\n')
    tex = tex.to_latex(column_format = 'l')
    with open(os.path.join('tables', 'trainingresults.tex'), 'w') as f:
        f.write(tex)
    return df, tex


# def latex_val(v, digits):
#     return r"\boldsymbol{{{:.{}f}}}".format(v, digits)
#         elif sort.index(v)==1:
#             return r"\underline{{{:.{}f}}}".format(v, digits)
#         else:
#             return _v

# def latex_std(s, digits):
#     return r"(\pm {0:.{}f})".format(s, digits)

def latex_col(vals, std = None, digits = 2, std_digits = None):
    sort = vals.sort_values(ascending = False).tolist()
    out = []
    if std is not None:
        if std_digits is None:
            std_digits = digits
        for v, s in zip(vals, std):
            if sort.index(v) == 0:
                it = r"$\boldsymbol{{{:.{}f}}} \ (\pm {:.{}f})$".format(v, digits, s, std_digits)
            elif sort.index(v) == 1:
                it = r"$\underline{{{:.{}f}}} \ (\pm {:.{}f})$".format(v, digits, s, std_digits)
            else:
                it = r"${:.{}f} \ (\pm {:.{}f})$".format(v, digits, s, std_digits)
            out.append(it)
        return out
    else:
        for v in enumerate(vals):
            if sort.index(v) == 0:
                it = r"$\boldsymbol{{{:.{}f}}}$".format(v, digits)
            elif sort.index(v) == 1:
                it = r"$\underline{{{:.{}f}}}$".format(v, digits)
            else:
                it = r"${:.{}f}$".format(v, digits)
            out.append(it)
        return out


def evaluationresults():
    dRC = pd.read_csv('results/RC.csv', index_col=0)
    dHA = pd.read_csv('results/HA.csv', index_col=0)
    dRC['metric'] = 'RC'
    dHA['metric'] = 'HA'
    df = pd.concat([dRC, dHA], axis = 0).reset_index(drop = True)
    out =[]
    for k, source in enumerate(['scientsbank', 'beetle']):
        dfs = []
        for method in __attr_methods__:
            for metric in ['HA', 'RC']:
                _df = df[(df['source']==source)&(df['metric']==metric)]
                col = metric + '_' + method
                vals = _df[_df['metric']==metric][method]
                col_vals = latex_col(vals, digits = 2)
                token_types = _df[_df['metric']==metric]['token_types']
                model_names =  _df[_df['metric']==metric]['model_name']
                _df = pd.DataFrame({
                    'model_name': model_names,
                    'token_types': token_types})
                _df[col] = [str(v) for v in col_vals]
                dfs.append(_df)
        _df = reduce(lambda x, y: pd.merge(x, y, on =  ['model_name', 'token_types']), dfs)
        _df = _df.sort_values('model_name', key = to_model_index)
        _df.reset_index(drop = True)
        _df.set_index(['model_name', 'token_types'])
        out.append(_df)
        with open(os.path.join('tables', f'evaluations_{source}.tex'), 'w') as f:
            f.write(_df.to_latex(index = False))
    return out
# def evaluationresults(aggr = 'L2', source = 'scientsbank'):
#     df = pd.read_csv('results/evaluationresults.csv', index_col = 0)
#     df = df[df['source']==source]
#     df = df[['model_index','model', 'source', 'token_types', 'metric'] +
#             [a for a in __attr_aggr__ if a.endswith(aggr)]]
#     df = df.rename(columns = {a: a.replace('_' + aggr, '') for a in __attr_aggr__})
#     dfs = []
#     for metric in ['DC', 'RC', 'HA']:
#         _df = df[df['metric'] == metric]
#         _df = _df.set_index(['model_index','model', 'token_types'])
#         _df = _df.drop(columns = ['metric', 'source'])
#         _df.columns = [metric + '_' + col for col in _df.columns]
#         if metric in ['DC', 'RC']:
#             _df = 0.5*(_df + 1)
#         _df = _df.round(2)
#         dfs.append(_df)
#     df = pd.concat(dfs, axis = 1)
#     df.index = df.index.droplevel()
#     df = df[[e + '_' + a for a in __attr_methods__ for e in ['DC', 'RC', 'HA'] ]]
#     with open(os.path.join('tables', source + '_evaluations_' + aggr + '.tex'), 'w') as f:
#         f.write(df.to_latex())
#     return df


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


# if __name__=='__main__':
#     groups, df = f1_table()
#     print(df)
#     #save_table(df, 'trainingresults')
