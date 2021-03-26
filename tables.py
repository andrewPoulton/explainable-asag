import pandas as pd


def f1_table():
    df_results = pd.read_csv('results/experiment_info.csv')
    dfs = []
    groups = ['scientsbank', 'scientsbank-token_types', 'beetle', 'beetle-token_types']
    for group in groups:
        dfs_group = [df_results[df_results['group'] == group + j][['name', 'max_f1']].set_index('name') for j in ['-1','-2','-3']]
        df_group = pd.concat(dfs_group, axis = 1)
        df_group['mean_f1'] = df_group.mean(axis = 1)
        df_group['std_f1'] = df_group.std(axis = 1)
        df_group.columns = [group + '_' + col for col in df_group.columns]
        dfs.append(df_group)
    df = pd.concat(dfs, axis = 1)
    df = df.round(2)
    for group in groups:
        df[group] = [f"${m} \pm {s}$" if m != 0 else '' for m,s in  zip(df[group + '_mean_f1'],df[group + '_std_f1'])]
    return groups, df

def save_table(df, table_name):
    df.to_csv(os.path.join('tables', table_name + '.csv'))
    with open(os.path.join('tables', table_name + '.tex'), 'w') as f:
        f.write(df.to_latex(column_format = 'r'))


if __name__=='__main__':
    groups, df = f1_table()
    print(df)
    #save_table(df, 'trainingresults')
