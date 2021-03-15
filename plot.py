import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from evaluation import (
    get_tokenizer,
    compute_golden_saliency_vector,
    compute_student_answer_word_saliency_vector,
    read_annotation,
    read_attribution,
    read_test_data,
    DATA_FILE,
    ATTRIBUTION_DIR,
    ANNOTATION_DIR
)
from weasyprint import HTML, CSS

TEST_annotation_file_names = [
    'sebas_beetle_unseen_answers_66.annotation'
]

TEST_attribution_file_names =[
    'albert-base_beetle_37zy0hai_GradientShap.pkl',
    'albert-base_beetle_37zy0hai_InputXGradient.pkl',
    'albert-base_beetle_37zy0hai_Saliency.pkl',
    'bert-base_beetle_1fkyddwv_GradientShap.pkl',
    'bert-base_beetle_1fkyddwv_InputXGradient.pkl',
    'bert-base_beetle_1fkyddwv_Saliency.pkl',
    'bert-large_beetle_9c9fsnl5_GradientShap.pkl',
    'bert-large_beetle_9c9fsnl5_InputXGradient.pkl',
 #   'bert-large_beetle_9c9fsnl5_IntegratedGradients.pkl',
    'bert-large_beetle_9c9fsnl5_Saliency.pkl',
    'distilbert-base_beetle_2sbg632j_GradientShap.pkl',
    'distilbert-base_beetle_2sbg632j_InputXGradient.pkl',
   # 'distilbert-base_beetle_2sbg632j_IntegratedGradients.pkl',
    'distilbert-base_beetle_2sbg632j_Saliency.pkl',
    # 'roberta-base_beetle_2owzw8vf_GradientShap.pkl',
    # 'roberta-base_beetle_2owzw8vf_InputXGradient.pkl',
    # 'roberta-base_beetle_2owzw8vf_IntegratedGradients.pkl',
    # 'roberta-base_beetle_2owzw8vf_Saliency.pkl',
    # 'roberta-large_beetle_xpdm71ti_GradientShap.pkl',
    # 'roberta-large_beetle_xpdm71ti_InputXGradient.pkl',
    # 'roberta-large_beetle_xpdm71ti_IntegratedGradients.pkl',
    # 'roberta-large_beetle_xpdm71ti_Saliency.pkl',
    # 'distilroberta_beetle_2nxwh65v_GradientShap.pkl',
    # 'distilroberta_beetle_2nxwh65v_InputXGradient.pkl',
    # 'distilroberta_beetle_2nxwh65v_IntegratedGradients.pkl',
    # 'distilroberta_beetle_2nxwh65v_Saliency.pkl',
    'bert-base-squad2_beetle_307twu8l_GradientShap.pkl',
    'bert-base-squad2_beetle_307twu8l_InputXGradient.pkl',
   # 'bert-base-squad2_beetle_307twu8l_IntegratedGradients.pkl',
    'bert-base-squad2_beetle_307twu8l_Saliency.pkl'
]


def get_human_df(annotations, student_answer):
    df  = pd.DataFrame(columns = ['attribution_method', 'model'] + student_answers.split(), index = range(len(annotations)))
    for i, a in enumerate(annotations):
        anno = a['annotation']
        saliency = compute_golden_saliency_vector(anno, student_answers)
        df.iloc[i] = ['Human Rationale', a['annotator']] + list(saliency)
    # make sure saliency is float
    for word in student_answer.split():
        df[word] = df[word].astype(float)
    # make annotators annonamous
    df['model'] = df['model'].apply(lambda x: 'human-' + str(df['model'].unique().tolist().index(x)))
    df = df.set_index(['attribution_method','model']).sort_index()
    return df

def get_ai_df(attributions, aggr, q_id, data_row):
    df  = pd.DataFrame(columns = ['attribution_method', 'model'] + student_answers.split(), index = range(len(attributions)))
    for i, a in enumerate(attributions):
        tokenizer = get_tokenizer(a['model'])
        df_attr = a['df'].set_index('question_id')
        attr = df_attr.loc[q_id, 'attr_' + aggr].astype(float)
        saliency = compute_student_answer_word_saliency_vector(attr, data_row, tokenizer)
        df.iloc[i] = [a['attribution_method'], a['model']] + list(saliency)
    df = df.set_index(['attribution_method','model']).sort_index()
    return df


def compute_saliency_heatmap_data(annotation_file_names, attribution_file_names, aggr = 'L2'):
    annotations = [read_annotation(ANNOTATION_DIR, file_name) for file_name in annotation_file_names]
    attributions = [read_attribution(ATTRIBUTION_DIR, file_name, attr_is_pred = True) for file_name in attribution_file_names]
    assert len(set(a['question_id'] for a in annotations)) <= 1, 'All annotations should be of same input'
    assert len(set([a['source'] for a in annotations] + [a['source'] for a in attributions])) <= 1, 'All attributions and annotations should be from same data source'
    q_id = int(annotations[0]['question_id'])
    df = read_test_data(annotations[0]['source'])
    # Set pred class to attr class and index of attribution dataframes to question_id
    for a in attributions:
        a['df'].index = df.index
    data_row = df.loc[q_id]
    df_human = get_human_df(annotations, data_row['student_answer'])
    df_ai =  get_ai_df(attributions, aggr, q_id, data_row)
    df_plot = pd.concat([df_human, df_ai], axis = 0)
    return df_plot,  data_row

def plot_saliency_heatmap_as_style_df(df, input_data):
    df = df.set_index(['attribution_method','model'])
    cm = sns.light_palette((260, 75, 60), input="husl", as_cmap=True)
    with pd.option_context('display.precision', 2):
        df = df.astype('Float64')
        df_ai = df.loc[df.index[1:]].sort_index()
        df_human = df.loc[df.index[0:1]]
        df = df_human.append(df_ai)
        df.index.names = [None,None]
        df_styled = df.style.background_gradient(cmap =cm).set_properties(**{'font-size': '12px'})
    html = HTML(string=df_styled.render())
    html.write_png('plots/saliency_heatmap_styled_df.png')
    pass


def plot_saliency_heatmap(df, input_data):
    df = df.set_index(['attribution_method','model']).astype(float)
    plt.figure(figsize=(16,9))
    cm = sns.light_palette((260, 75, 60), input="husl", as_cmap=True)
    sns.heatmap(df.round(2), annot = True, cmap = cm)
    plt.savefig('plots/saliency_heatmap.png')





if __name__ == "__main__":
    df, input_data = compute_saliency_heatmap_data(TEST_annotation_file_names, TEST_attribution_file_names)
    plot = plot_saliency_heatmap(df, input_data)
