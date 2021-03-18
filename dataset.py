from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace
# from transformers.tokenization_bert import BertTokenizer, BasicTokenizer
from collections import OrderedDict, Counter
from itertools import combinations, permutations
from random import choice
from typing import List, Dict
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DataLoader
import xml.etree.ElementTree as et
import pandas as pd
import torch
import re
import configuration

def pad_tensor_batch(tensors, pad_token = 0):
    max_length = max([t.size(0) for t in tensors])
    batch = torch.zeros((len(tensors), max_length)).long()
    if pad_token > 0:
        batch.fill_(pad_token)
    for i, tensor in enumerate(tensors):
        batch[i, :tensor.size(0)] = tensor
    return batch

class Batch(SimpleNamespace):
    def __init__(self, **kwargs):
        super(Batch, self).__init__(**kwargs)

    def cuda(self):
        atts = self.__dict__
        for att, val in atts.items():
            try:
                self.__dict__[att] = val.cuda()
            except AttributeError:
                pass

    def cpu(self):
        atts = self.__dict__
        for att, val in atts.items():
            try:
                self.__dict__[att] = val.cpu()
            except AttributeError:
                pass

    def __contains__(self, item):
        return item in self.__dict__

    def generate_mask(self):
        assert "input" in self
        return torch.where(self.input.eq(0), self.input, torch.ones_like(self.input))


class SemEvalDataset(Dataset):
    def __init__(self, data_file = 'data/flat_semeval5way_train.csv', vocab_file = 'bert-base-uncased',  num_labels = 2, train_percent = 100):
        self.data = pd.read_csv(data_file)
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, lowercase=True)
        self.label_map = {'correct':0,
                        'contradictory':1,
                        'partially_correct_incomplete':2,
                        'irrelevant':3,
                        'non_domain':4}
        self._3way_labels = lambda x: 2 if x >=2 else x
        self._2way_labels = lambda x: 1 if x >=1 else x
        self.progress_encode()
        self._data = self.data.copy()
        self.test = 'test' in data_file
        self.source = ''
        self.train_percent = train_percent
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def set_question(self, problem_id):
        new_data = self._data[self._data.problem_index == problem_id]                     
        self.data = new_data
        

    def to_val_mode(self, source, origin):
        data = self._data[self._data.source == source]
        data = data[data.origin.str.contains(origin)]
        self.data = data
        if self.train_percent < 100:
            num_egs = int(self.train_percent*0.01*len(self.data))
            self.data = self.data.iloc[:num_egs]

    # what is the function of set_data_source and why select train_percent here?
    def set_data_source(self, source):
        data = self._data[self._data.source == source]
        self.data = data
        self.source = source
        if self.train_percent < 100:
            num_egs = int(self.train_percent*0.01*len(self.data))
            self.data = self.data.iloc[:num_egs]

    def progress_encode(self):
        if self.num_labels == 2:
            label_map = self._2way_labels
        elif self.num_labels == 3:
            label_map = self._3way_labels
        else:
            label_map = lambda x: x
        encoded_text = [None] * len(self.data)
        token_types = [None] * len(self.data)
        labels = [None] * len(self.data)
        og_labels = [None] * len(self.data)
        is_rand = [] 
        seen_questions = {}
        seen_idx = 0
        problem_index = [None] * len(self.data)
        for i in tqdm(range(len(self.data)), desc="Encoding text"):
            row = self.data.iloc[i]
            question_tokens = self.tokenizer.encode(row['question_text'])
            reference_tokens = self.tokenizer.encode(row['reference_answers'])
            student_tokens = self.tokenizer.encode(row['student_answers'])

            token_type_ids = [0] + [1]*(len(question_tokens)-1) + [2]*(len(reference_tokens)-1) + [3]*(len(student_tokens)-1)
            tokens = question_tokens + reference_tokens[1:] + student_tokens[1:]
            encoded_text[i] = torch.Tensor(tokens).long()
            token_types[i] = torch.Tensor(token_type_ids).long()
            labels[i] = label_map(self.label_map[row.label])
            og_labels[i] = row.label
            if row['question_text'] in seen_questions:
                problem_index[i] = seen_questions[row['question_text']]
            else:
                seen_questions[row['question_text']] = seen_idx
                problem_index[i] = seen_questions[row['question_text']]
                seen_idx += 1
        self.data['encoded_text'] = encoded_text
        self.data['token_types'] = token_types
        self.data['labels'] = labels
        self.data['problem_index'] = problem_index
        self.data['original_label'] = og_labels

    def get_row(self, i):
        return self.data.loc[i]
    
    @staticmethod
    def collater(batch):
        input_ids = [b.encoded_text for b in batch]
        token_type_ids = [b.token_types for b in batch]
        labels = [b.labels for b in batch]
        data = {'input':pad_tensor_batch(input_ids),
                'token_type_ids':pad_tensor_batch(token_type_ids),
                'labels': torch.Tensor(labels).long()}
        return Batch(**data)



def dataloader(
        data_file = '',
        data_source = '',
        vocab_file = '',
        num_labels = 2,
        train_percent = 100,
        batch_size = 0,
        drop_last = False,
        num_workers = 0,
        data_val_origin = 'answer',
        val_mode = False)
    if val_mode and 'test' not in data_file:
        data_file = data_file.replace('train', 'test')
    # now we define the dataloader
    data = SemEvalDataset(data_file = data_file, vocab_file = vocab_file, train_percent = train_percent)
    data.set_data_source(data_source)
    print(f"Data loaded from {data_file} with {data.data.shape[0]} lines.")
    if val_mode:
        data.to_val_mode(data_source, 'answer')
    sampler = SequentialSampler(data) if val_mode else RandomSampler(data)
    batch_sampler = BatchSampler(sampler, batch_size = batch_size, drop_last=drop_last)
    loader = DataLoader(data, batch_sampler=batch_sampler, collate_fn=data.collater, num_workers = num_workers)
    return loader
