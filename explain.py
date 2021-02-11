## Implement:
## TODO: inputxgrad
## TODO: inputXgradient
## WAIT: occlusion
## TODO: saliency
## TODO: integrated gradients


import fire
import numpy as np

import torch
import torch.nn as nn

import transformers

from captum.attr import (
    IntegratedGradients,
    InputXGradient)

import dataset

def construct_ref_ids(input_ids, cls_token_id = 101, sep_token_id = 102, ref_token_id = 0):
    ref_ids = torch.zeros_like(input_ids)
    if ref_token_id !=0:
        ref_ids.fill_(ref_token_id)
    ref_ids[input_ids==cls_token_id] = cls_token_id
    ref_ids[input_ids==sep_token_id] = sep_token_id
    return ref_ids



# def explainer(model, attribution_method = IntegratedGradients):
#     return IntegratedGradients(model)

def explain_validation_data():
    val_dataloader = dataset.dataloader(data_file = 'data/mini_test.csv', val_mode = True, batch_size = 1, num_workers = 0)
    # val_dataloader = dataset.dataloader(
    #     val_mode = True,
    #     data_file = config.val_data,
    #     data_source = config.data_source,
    #     vocab_file = config.model_name,
    #     num_labels = config.num_labels,
    #     train_percent = config.val_percent,
    #     batch_size = 1,
    #     drop_last = config.drop_last,
    #     num_workers = config.num_workers)
    model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)
    # model = torch.load(path_to_model)
    model.eval()

    explainer =  IntegratedGradients(model)
    attr = explainer.attribute(inputs = b.input, baselines = ref_ids, target = 0)


if __name__=='__main__':
    fire.Fire(explain_validation_data)
