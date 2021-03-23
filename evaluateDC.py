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

__CUDA__ = torch.cuda.is_available()
__ANNOTATION_DIR__ = 'annotator/annotations'
__ATTRIBUTION_DIR__ = 'attributions'
__RESULTS_DIR__ = 'results'

def evaluate_dataset_consistency(*attribution_files):
    dc_list = []
    for attr_file in attribution_files:
        attr_data = AttributionData(attr_file)
        dc = compute_dataset_consistency(attr_data, cuda = __CUDA__)
        dc_list.append(dc)
    return pd.DataFrame.from_records(dc_list)

def evaluateDC(attributions_dir):
    if attributions_dir.endswith('/'):
        attributions_dir = attributions_dir[:-1]
    attr_files = [os.path.join(attributions_dir, f) for f in os.listdir(attributions_dir) if f.endswith('.pkl')]
    df = evaluate_dataset_consistency(*attr_files)
    directory = '/'.join(attributions_dir.split('/')[:-1])
    directory = os.path.join(__RESULTS_DIR__,directory)
    filename = attributions_dir.split('/')[-1]
    filename += '_DC.csv'
    if not os.path.isdir(directory):
        os.mkdir(directory)
    df.to_csv(os.path.join(directory, filename))

if __name__ == '__main__':
    fire.Fire(evaluateDC)
