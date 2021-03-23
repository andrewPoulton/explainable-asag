import torch
import fire
import os
import json
import random
import pickle
import gc
import wandb
import dataset
from tqdm import tqdm
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score#    import wandb
from modelhandling import get_sub_module

def grad_norm(model):
    return sum(p.grad.pow(2).sum() if p.grad is not None else torch.tensor(0.) for p in model.parameters())**.5

def compute_logits(model, batch, token_types):
    mask = batch.generate_mask()
    if token_types:
        logits = model(input_ids = batch.input, attention_mask = mask, token_type_ids = batch.token_type_ids)
    else:
        logits = model(input_ids = batch.input, attention_mask = mask)
    logits = logits[0]
    return logits

def train_epoch(loader, model, optimizer, lr_scheduler, num_labels, cuda, log = False, token_types = False):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    with tqdm(total=len(loader.batch_sampler)) as pbar:
        epoch_loss = 0.
        for i, batch in enumerate(loader):
            if cuda:
                batch.cuda()
            optimizer.zero_grad()
            logits = compute_logits(model, batch, token_types)
            # import pdb; pdb.set_trace()
            loss = loss_fn(logits.view(-1, num_labels), batch.labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            lr_scheduler.step()

           
            if batch.labels.size(0)>1:
                acc = accuracy_score(batch.labels.cpu(), logits.cpu().detach().argmax(-1).squeeze())
            else:
                acc = 0.
            # if torch._np.isnan(loss.item()):
            # import pdb; pdb.set_trace()
            epoch_loss += loss.item()
            # if i % config.log_interval == 0:
            if log:
                wandb.log({"Train Accuracy": acc, "Train Loss": loss.item(), "Learning Rate": optimizer.param_groups[0]['lr']})
            if log and token_types:
                 wandb.log({"Word embedding grad norm":grad_norm(get_sub_module(model, "word_embeddings")).item(),
                           "Token type embedding grad norm":grad_norm(get_sub_module(model, "token_type_embeddings")).item()})
            pbar.set_description(f'global_step: {lr_scheduler.last_epoch}| loss: {loss.item():.4f}| acc: {acc*100:.1f}%| epoch_av_loss: {epoch_loss/(i+1):.4f} |')
            pbar.update(1)
        #  move stuff off GPU
        batch.cpu()
        logits = logits.cpu().detach().argmax(-1).squeeze()
        return epoch_loss/(i+1)


def metrics(predictions, y_true, metric_params):
    precision = precision_score(y_true, predictions, **metric_params)
    recall = recall_score(y_true, predictions, **metric_params)
    f1 = f1_score(y_true, predictions, **metric_params)
    accuracy = accuracy_score(y_true, predictions)
    return precision, recall, f1, accuracy

@torch.no_grad()
def val_loop(model, loader, cuda,  token_types = False):
    model.eval()
    # batches = list(loader)
    preds = []
    true_labels = []
    with tqdm(total= len(loader.batch_sampler)) as pbar:
        for i,batch in enumerate(loader):
            if cuda:
                batch.cuda()
            logits = compute_logits(model, batch, token_types)
            preds.append(logits.argmax(-1).squeeze().cpu())
            true_labels.append(batch.labels.cpu())
            pbar.update(1)
    preds = torch.cat(preds)
    y_true = torch.cat(true_labels)
    model.train()
    metric_params_weighted = {'average':'weighted', 'labels':list(range(model.config.num_labels))}
    metric_params_macro =  {'average':'macro', 'labels':list(range(model.config.num_labels))}
    return metrics(preds, y_true, metric_params_weighted), metrics(preds, y_true, metric_params_macro)
