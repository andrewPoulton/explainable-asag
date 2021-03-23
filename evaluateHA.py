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
__ANNOTATION_DIR__ = 'annotator/annotations'
__ATTRIBUTION_DIR__ = 'attributions'
__RESULTS_DIR__ = 'results'

def evaluate_human_agreement(annotation_dir, *attr_files):
    annotation_data = AnnotationData(annotation_dir)
    annotation_data.set_annotator('sebas')
    ha_list = []
    for attribution_file in attr_files:
        attribution_data = AttributionData(attribution_file)
        annotation_data.set_source(attribution_data.source)
        ha = compute_human_agreement(attribution_data, annotation_data)
        ha.update({'run_id': attribution_data.run_id, 'attr_file': attribution_file})
        ha_list.append(ha)
    return pd.DataFrame.from_records(ha_list)

def evaluateHA(annotation_dir, attributions_dir):
    if attributions_dir.endswith('/'):
        attributions_dir = attributions_dir[:-1]
    attr_files = [os.path.join(attributions_dir, f) for f in os.listdir(attributions_dir) if f.endswith('.pkl')]
    df = evaluate_human_agreement(annotation_dir, *attr_files)
    directory = '/'.join(attributions_dir.split('/')[:-1])
    directory = os.path.join(__RESULTS_DIR__,directory)
    filename = attributions_dir.split('/')[-1]
    filename += '_HA.csv'
    if not os.path.isdir(directory):
        os.mkdir(directory)
    df.to_csv(os.path.join(directory, filename))

if __name__ == '__main__':
    fire.Fire(evaluateHA)
