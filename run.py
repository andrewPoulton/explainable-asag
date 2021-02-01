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
   failing = [
      "albert-base-squad2",
### TODO: Throws ERROR on loading tokenizer:
### ValueError: Couldn't instantiate the backend tokenizer from one of:
### (1) a `tokenizers` library serialization file,
### (2) a slow tokenizer instance to convert or (3) an equivalent slow tokenizer class to instantiate and convert.
### You need to have sentencepiece installed to convert a slow tokenizer to a fast one.
      "roberta-large-stsb",
      "distilroberta-base-stsb"
### TODO: Throw ERROR on computing loss in training. Something to do with non-matching batch sizes.
### with batch_size = 4 I got
### ValueError: Expected input batch_size (1) to match target batch_size (2).
### with batch_size = 1 I got
### RuntimeError: shape '[-1, 2]' is invalid for input of size 1
   ]
   for  experiment in [e for e in  configs.list_experiments() if e not in failing]:
        config = configs.load(experiment)
        data_train = dataset.SemEvalDataset(data_file = 'data/flat_semeval5way_train.csv', vocab_file = config.model_path,
                                            train_percent = 0.001) # for rapid testing... change back to 100

        sampler = RandomSampler(data_train)
        batch_sampler = BatchSampler(sampler, batch_size = 1, drop_last=False)
        loader = DataLoader(data_train, batch_sampler=batch_sampler, collate_fn=data_train.collater,
                            num_workers = 0) # Change to i > 0 for multi-processing, I ran into error on my mac

        model = AutoModelForSequenceClassification.from_pretrained(config.model_path)

        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 1, 1)
        cuda  = False

        num_labels= 2
        total_steps = 2
        training.train_epoch(loader, model, optimizer, lr_scheduler, num_labels, total_steps, cuda)


if __name__ == '__main__':
    fire.Fire(run)
