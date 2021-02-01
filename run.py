import dataset
import training
import configs
import utils
import transformers
from transformers import AutoModelForSequenceClassification
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DataLoader
from types import SimpleNamespace
import fire
import torch
from pathlib import Path


import os
import utils
import fire


def run():
    ## init loader
    config = configs.load('albert-base')
    data_train = dataset.SemEvalDataset('data/flat_semeval5way_train.csv', config.model_path, train_percent = 100)
    sampler = RandomSampler(data_train)
    batch_sampler = BatchSampler(sampler, batch_size = 2, drop_last=False)
    loader = DataLoader(data_train, batch_sampler=batch_sampler, collate_fn=data_train.collater,  num_workers = 0)

    ## init model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

    ## other parameters
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 1, 4)
    cuda  = False

    ## mock config
    config = SimpleNamespace(**{'num_labels':2, 'total_steps': 24*1024})
    training.train_epoch(loader, model, optimizer, lr_scheduler, config, cuda)


if __name__ == '__main__':
    fire.Fire(run)
