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
import os
import itertools


from torch.utils.data.sampler import Sampler

__TRAIN_DATA__ = os.path.join('data', 'flat_semeval5way_train.csv')
__TEST_DATA__ = os.path.join('data', 'flat_semeval5way_test.csv')

def get_word_structure(tokenizer, text, offset = 0, **kwargs):
    encoded_plus = tokenizer.encode_plus(text)
    char_to_token = [encoded_plus.char_to_token(i) for i, c in enumerate(text)]
    char_to_token = [None] + char_to_token + [None]
    if offset > 0:
        char_to_token = [t + int(offset) if isinstance(t, int) else t for t in char_to_token]
    none_pos = [i for i,t in enumerate(char_to_token) if t is None]
    # words are the tokens between None's in char_to_token
    word_structure = [list(set(char_to_token[i+1:j])) for i,j in zip(none_pos[:-1],none_pos[1:])]
    return word_structure


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
    def __init__(self, data_file = __TRAIN_DATA__, vocab_file = 'bert-base-uncased',  num_labels = 2, train_percent = 100):
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

    def get_instance(self, idx, word_structure =  False):
        row =  self._data.loc[int(idx)].to_dict()
        if word_structure:
            q_ws = get_word_structure(self.tokenizer, row['question_text'], offset = 0)
            ref_ws = get_word_structure(self.tokenizer, row['reference_answers'],
                                        offset = (row['token_types']<2).count_nonzero()-1)
            st_ws = get_word_structure(self.tokenizer, row['student_answers'],
                                        offset = (row['token_types']<3).count_nonzero()-1)
            row.update({'word_structure': {'question_text': q_ws, 'reference_answer': ref_ws, 'student_answer': st_ws}})
        return row

    @staticmethod
    def collater(batch):
        input_ids = [b.encoded_text for b in batch]
        token_type_ids = [b.token_types for b in batch]
        labels = [b.labels for b in batch]
        instance_ids = [b.name for b in batch]
        data = {'input':pad_tensor_batch(input_ids),
                'token_type_ids':pad_tensor_batch(token_type_ids),
                'labels': torch.Tensor(labels).long(),
                'instance': torch.Tensor(instance_ids).long()}
        return Batch(**data)



class PairSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield mini-batches of all pairs of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object

    Example:
        >>> list(PairSampler(SequentialSampler(range(3))))
        [[0, 1],[0,2],[1,2]]
    """

    def __init__(self, sampler: Sampler[int], **kwargs) -> None:
        self.sampler = sampler
        self.kwargs = kwargs

    def __iter__(self):
        if 'within' in self.kwargs or 'between' in self.kwargs:
            sampler_list = list(self.sampler)
            if 'within' in self.kwargs:
                labels = self.kwargs['within']
                assert len(labels) == len(self.sampler), 'Labels of groups to sample within need to have length of sampler'
                groups = [[i for k,i in enumerate(sampler_list) if labels[k]==label] for label in list(set(labels))]
                group_iters = [itertools.combinations(group,2) for group in groups]
                within_pairs =  itertools.chain(*group_iters)
            else:
                within_pairs = iter([])

            if 'between' in self.kwargs:
                labels = self.kwargs['between']
                assert len(labels) == len(self.sampler), 'Labels of groups to sample between need to have length of sampler'
                groups = [[i for k,i in enumerate(sampler_list) if labels[k]==label] for label in list(set(labels))]
                group_sampler = iter(choice(g) for g in groups)
                between_pairs = itertools.combinations(group_sampler,2)
            else:
                between_pairs = iter([])
            pairs = itertools.chain(within_pairs, between_pairs)
        else:
            pairs = itertools.combinations(self.sampler,2)

        for pair in pairs:
            yield list(pair)

    def __len__(self):
        if 'within' in self.kwargs or 'between' in self.kwargs:
            # This is pretty dumb, but easiest for now
            return len(list(self.__iter__()))
        else:
            return len(self.sampler)*(len(self.sampler)-1)//2

def dataloader(data_file = None,
        data_source = None,
        vocab_file =  None,
        num_labels = 2,
        train_percent = 100,
        batch_size = 1,
        drop_last = False,
        num_workers = 0,
        data_val_origin = 'unseen_answers',
        val_mode = False
        ):

    if val_mode and 'test' not in data_file:
        data_file = data_file.replace('train', 'test')
    data = SemEvalDataset(data_file = data_file, vocab_file = vocab_file, train_percent = train_percent)
    data.set_data_source(data_source)
    if val_mode:
        data.to_val_mode(data_source, 'answer')
    print(f"Data loaded from {data_file} with {data.data.shape[0]} lines.")
    sampler = SequentialSampler(data) if val_mode else RandomSampler(data)
    batch_sampler =BatchSampler(sampler, batch_size = batch_size, drop_last=drop_last)
    loader = DataLoader(data, batch_sampler=batch_sampler, collate_fn=data.collater, num_workers = num_workers)
    return loader


def pairdataloader(
        data_file = os.path.join('data', 'flat_semeval5way_test.csv'),
        data_source = 'scientsbank',
        vocab_file =  'bert-base-uncased',
        num_labels = 2,
        train_percent = 100,
        num_workers = 8 if torch.cuda.is_available() else 0,
        data_val_origin = 'unseen_answers',
        val_mode = True,
        within_questions = False,
        between_questions = False,
        ):
    if val_mode and 'test' not in data_file:
        data_file = data_file.replace('train', 'test')
    data = SemEvalDataset(data_file = data_file, vocab_file = vocab_file, train_percent = train_percent)
    data.set_data_source(data_source)
    if val_mode:
        data.to_val_mode(data_source, 'answer')
    print(f"Data loaded from {data_file} with {data.data.shape[0]} lines.")
    sampler = SequentialSampler(data)
    kwargs = {}
    if within_questions:
        kwargs.update({'within': data.data['problem_index'].tolist()})
    if between_questions:
        kwargs.update({'between': data.data['problem_index'].tolist()})
    pair_sampler = PairSampler(sampler, **kwargs)
    loader = DataLoader(data, batch_sampler=pair_sampler, collate_fn=data.collater, num_workers = num_workers)
    return loader
