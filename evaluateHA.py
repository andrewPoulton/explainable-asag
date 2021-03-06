import fire
import os
import pandas as pd
from evaluation import (
    AnnotationData,
    AttributionData,
    compute_human_agreement,
    __RESULTS_DIR__
)
import torch
from wandbinteraction import get_runs
import re

__CUDA__ = torch.cuda.is_available()

def evaluateHA(annotation_dir, attributions_dir):
    os.makedirs(__RESULTS_DIR__, exist_ok=True)
    attr_files = [os.path.join(attributions_dir, f) for f in os.listdir(attributions_dir) if f.endswith('.pkl')]
    path_pieces =  os.path.normpath(attributions_dir).split(os.sep)
    group = re.sub('\-[0-9]$', '', path_pieces[-1])
    filepath = os.path.join(__RESULTS_DIR__, group + '_HA.csv')
    annotation_data = AnnotationData(annotation_dir)
    annotation_data.set_annotator('sebas')
    ha_list = []
    for attribution_file in attr_files:
        attribution_data = AttributionData(attribution_file)
        annotation_data.set_source(attribution_data.source)
        ha = compute_human_agreement(attribution_data, annotation_data)
        ha.update({'run_id': attribution_data.run_id, 'attr_file': attribution_file})
        ha_list.append(ha)
    df =  pd.DataFrame.from_records(ha_list)
    df.to_csv(filepath)

if __name__ == '__main__':
    fire.Fire(evaluateHA)
