import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from evaluation import (
    AttributionData,
    AnnotationData,
    scale_to_unit_interval
)
from dataset import SemEvalDataset
import torch


def make_figure():
    i = 14
    a = AttributionData('attributions/beetle-1/2sym8yv8.pkl')
    a.set_attr_class('pred')
    attr = a.df.set_index('instance_id').loc[i]
    data = a.get_dataset()
    instance = data.get_instance(i)
    Q = instance['encoded_text'][instance['token_types']==1]
    R = instance['encoded_text'][instance['token_types']==2]
    S = instance['encoded_text'][instance['token_types']==3]
    print(scale_to_unit_interval(attr['IntegratedGradients']['L2'], 'L2'))
