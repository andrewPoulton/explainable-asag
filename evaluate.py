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

def evaluate_human_agreement(annotations_dir, *attr_files):
    annotation_data = AnnotationData(annotations_dir)
    annotation_data.set_annotator('sebas')
    ha_list = []
    for attribution_file in attr_files:
        attribution_data = AttributionData(attribution_file)
        annotation_data.set_source(attribution_data.source)
        ha = compute_human_agreement(attribution_data, annotation_data)
        ha.update({'run_id': attribution_data.run_id, 'attr_file': attribution_file})
        ha_list.append(ha)
    return pd.DataFrame.from_records(ha_list)


def evaluate_rationale_consistency(*pairs_of_attribution_files):
    rc_list = []
    for attr_file1, attr_file2 in pairs_of_attribution_files:
        attr_data1 = AttributionData(attr_file1)
        attr_data2 = AttributionData(attr_file2)
        if attr_data1.is_compatible(attr_data2):
            rc = compute_rationale_consistency(attr_data1, attr_data2, __CUDA__)
            rc.update({'run_id1':attr_data1.run_id, 'run_id2':attr_data2.run_id,
                       'attr_file1': attr_file1, 'attr_file2':attr_file2})
            rc_list.append(rc)
    return pd.DataFrame.from_records(rc_list)

def evaluate_dataset_consistency(*attribution_files):
    dc_list = []
    for attr_file in attribution_files:
        attr_data = AttributionData(attr_file)
        dc = compute_dataset_consistency(attr_data)
        dc_list.append(dc)
    return pd.DataFrame.from_records(dc_list)
def evaluate():
    # annotation_data = AnnotationData('annotator/annotations')
    # annotation_data.set_annotator('sebas')
    # df =  evaluate_human_agreement('annotator/annotations', 'explained/scientsbank-1/31ktjuah.pkl')
    # print('HS:',df)
    # df =  evaluate_rationale_consistency(['explained/scientsbank-1/ektq6s3e.pkl', 'explained/scientsbank-1/ektq6s3e.pkl'])
    # print('RC:', df)
    df =  evaluate_dataset_consistency('explained/scientsbank-1/ektq6s3e.pkl')
    print('DC', df)


if __name__ == '__main__':
    fire.Fire(evaluate)
