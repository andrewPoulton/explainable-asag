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



def train_epoch(loader, model, optimizer, lr_scheduler, config, cuda):
    loss_fn = torch.nn.CrossEntropyLoss()
    with tqdm(total=len(loader.batch_sampler)) as pbar:
        epoch_loss = 0.
        for i, batch in enumerate(loader):
            if cuda:
                batch.cuda()
            optimizer.zero_grad()
            mask = batch.generate_mask()
            logits = model(input_ids = batch.input, attention_mask = mask)
            logits = logits[0]
            # import pdb; pdb.set_trace()
            loss = loss_fn(logits.view(-1, config.num_labels), batch.labels.view(-1))
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
            # wandb.log({"Train Accuracy": acc, "Train Loss": loss.item(), "Gradient Norm": grad_norm(model).item(), "Learning Rate": optimizer.param_groups[0]['lr']})
            pbar.set_description(f'global_step: {lr_scheduler.last_epoch}| loss: {loss.item():.4f}| acc: {acc*100:.1f}%| epoch_av_loss: {epoch_loss/(i+1):.4f} |')
            pbar.update(1)
            if lr_scheduler.last_epoch > config.total_steps:
                break
        #  move stuff off GPU
        batch.cpu()
        logits = logits.cpu().detach().argmax(-1).squeeze()
        return epoch_loss/(i+1)