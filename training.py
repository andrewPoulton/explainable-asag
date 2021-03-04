import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import torch
    import dataset
    import fire
    import os
    import json
    import random
    import pickle
    from tqdm import tqdm
#    from utils import  load_config, init_model, configure_model
    from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DataLoader
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#    import wandb
    import gc
    import wandb


def get_sub_module(module, submodule_name):
    for attr_str in dir(module):
        if attr_str == submodule_name:
            return  getattr(module, attr_str)

    for n, ch in module.named_children():
        submodule = get_sub_module(ch, submodule_name)
        if submodule:
            return submodule

def grad_norm(model):
    return sum(p.grad.pow(2).sum() if p.grad is not None else torch.tensor(0.) for p in model.parameters())**.5

def train_epoch(loader, model, optimizer, lr_scheduler, num_labels, cuda, log = False, token_types = False):
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    with tqdm(total=len(loader.batch_sampler)) as pbar:
        epoch_loss = 0.
        for i, batch in enumerate(loader):
            if cuda:
                batch.cuda()
            optimizer.zero_grad()
            mask = batch.generate_mask()
            if token_types:
                logits = model(input_ids = batch.input, attention_mask = mask, token_type_ids = batch.token_type_ids)
            else:
                logits = model(input_ids = batch.input, attention_mask = mask)
            logits = logits[0]
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
