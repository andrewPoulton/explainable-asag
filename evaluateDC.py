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

__CUDA__ = torch.cuda.is_available()

def evaluate_dataset_consistency(*attribution_files, **kwargs):
    dc_list = []
    for attr_file in attribution_files:
        attr_data = AttributionData(attr_file)
        dc = compute_dataset_consistency(attr_data, **kwargs)
        dc_list.append(dc)
    return pd.DataFrame.from_records(dc_list)

def evaluateDC(attributions_dir, selection = False):
    attr_files = [os.path.join(attributions_dir, f) for f in os.listdir(attributions_dir) if f.endswith('.pkl')]
    path_pieces =  os.path.normpath(attributions_dir).split(os.sep)
    directory = os.path.join(__RESULTS_DIR__, path_pieces[-2])
    filename = path_pieces[-1]
    if selection:
        filename += '_DCselect.csv'
        kwargs = {'within_questions': True,
                  'between_questions': True}
    else:
        filename += '_DC.csv'
        kwargs = {}
    if not os.path.isdir(directory):
        os.mkdir(directory)
    df = evaluate_dataset_consistency(*attr_files, cuda = __CUDA__,**kwargs)
    df.to_csv(os.path.join(directory, filename))

if __name__ == '__main__':
    fire.Fire(evaluateDC)
