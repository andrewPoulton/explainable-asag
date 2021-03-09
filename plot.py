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
    'albert-base_beetle_37zy0hai_IntegratedGradients.pkl',
    'albert-base_beetle_37zy0hai_Saliency.pkl',
    'bert-base_beetle_1fkyddwv_GradientShap.pkl',
    'bert-base_beetle_1fkyddwv_InputXGradient.pkl',
    'bert-base_beetle_1fkyddwv_IntegratedGradients.pkl',
    'bert-base_beetle_1fkyddwv_Saliency.pkl',
    'bert-large_beetle_9c9fsnl5_GradientShap.pkl',
    'bert-large_beetle_9c9fsnl5_InputXGradient.pkl',
    'bert-large_beetle_9c9fsnl5_IntegratedGradients.pkl',
    'bert-large_beetle_9c9fsnl5_Saliency.pkl',
    'distilbert-base_beetle_2sbg632j_GradientShap.pkl',
    'distilbert-base_beetle_2sbg632j_InputXGradient.pkl',
    'distilbert-base_beetle_2sbg632j_IntegratedGradients.pkl',
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
    'bert-base-squad2_beetle_307twu8l_IntegratedGradients.pkl',
    'bert-base-squad2_beetle_307twu8l_Saliency.pkl'
]

def compute_saliency_heatmap_data(annotation_file_names, attribution_file_names, aggr = 'norm'):
    annotations = [read_annotation(ANNOTATION_DIR, file_name) for file_name in annotation_file_names]
    attributions = [read_attribution(ATTRIBUTION_DIR, file_name) for file_name in attribution_file_names]
    assert len(set(a['question_id'] for a in annotations)) <= 1, 'All annotations should be of same input'
    assert len(set([a['source'] for a in annotations] + [a['source'] for a in attributions])), 'All attributions and annotations should be from same data source'
    q_id = int(annotations[0]['question_id'])
    df = read_test_data(annotations[0]['source'])
    # Set pred class to attr class and index of attribution dataframes to question_id
    for a in attributions:
        a['df'] = a['df'][a['df']['pred'] == a['df']['attr_class']]
        a['df'].index = df.index
    data_row = df.loc[q_id]
    df_plot = pd.DataFrame(columns = ['attribution_method', 'model'] + data_row['student_answers'].split(), index = range(len(annotations) + len(attributions)))
    for i, a in enumerate(annotations):
        saliency = compute_golden_saliency_vector(a['annotation'], data_row['student_answers'])
        df_plot.iloc[i] = ['Human', a['annotator']] + list(saliency)
    for i, a in enumerate(attributions):
        tokenizer = get_tokenizer(a['model'])
        attr = a['df'].loc[q_id, 'attr_' + aggr].astype(float)
        text = tokenizer.convert_tokens_to_string(a['df'].loc[q_id, 'tokens'])
        saliency = compute_student_answer_word_saliency_vector(attr, data_row, tokenizer)
        df_plot.iloc[i+len(annotations)] = [a['attribution_method'], a['model']] + list(saliency)
    return df_plot,  data_row

def plot_saliency_heatmap(df, input_data):
    df = df.set_index(['attribution_method','model'])
    cm = sns.light_palette((260, 75, 60), input="husl", as_cmap=True)
    #cm = sns.diverging_palette(250, 5, as_cmap=True)
    with pd.option_context('display.precision', 2):
        df = df.astype('Float64')
        df_ai = df.loc[df.index[1:]].sort_index()
        df_human = df.loc[df.index[0:1]]
        df = df_human.append(df_ai)
        df.index.names = [None,None]
        df_styled = df.style.background_gradient(cmap =cm).set_properties(**{'font-size': '12px'})
    html = HTML(string=df_styled.render())
    html.write_png('plots/saliency_heatmap.png')
    pass


if __name__ == "__main__":
    df, input_data = compute_saliency_heatmap_data(TEST_annotation_file_names, TEST_attribution_file_names)
    plot = plot_saliency_heatmap(df, input_data)
