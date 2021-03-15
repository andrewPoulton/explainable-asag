import pandas as pd

def get_wandb_df():
    wandb_df = pd.read_csv('tables/experiments.csv', index_col = 0)
    return wandb_df

def get_group(source):
    wandb_df = get_wandb_df()
    if source == 'scientsbank':
        group = 'group'
    elif source =='beetle':
        group = 'beetlebeetle'
    else:
        error('Only allowed data sources: scientsbank and beetle')
    return group

def get_runs_models(source):
    df_wandb = get_wandb_df()
    group = get_group(source)
    dfs = [df_wandb[df_wandb['group'] == group + j][['run_id', 'name']].set_index('run_id') for j in ['-1','-2','-3']]
    for i, df in enumerate(dfs):
        df['group'] = source + '-' + str(i+1)
    df = pd.concat(dfs, axis = 0)
    for i, row in df.iterrows():
        print(row.name, '\t#', row['name'] ,row['group'])
    return df


def f1_table(source):
    df_wandb = get_wandb_df()
    group = get_group(source)
    dfs = [df_wandb[df_wandb['group'] == group + j][['name', 'max_f1']].set_index('name') for j in ['-1','-2','-3']]
    df = pd.concat(dfs, axis = 1)
    df['mean'] = df.mean(axis = 1)
    df['std'] = df.std(axis = 1)
    return df

if __name__=='__main__':
    df = f1_table('scientsbank')
    df.to_csv('tables/f1_scientsbank.csv')
    with open('tables/f1_scientsbank.tex', 'w') as f:
        f.write(df.to_latex())


    df = f1_table('beetle')
    df.to_csv('tables/f1_beetle.csv')
    with open('tables/f1_beetle.tex', 'w') as f:
        f.write(df.to_latex())
