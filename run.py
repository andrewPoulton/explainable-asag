import dataset
import training
import validation
import configuration
import transformers
import fire
import torch
import wandb
import gc
import os
from datetime import datetime
import sys
import json

### From: https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
def optimizer_to_cpu(optim):
    device = 'cpu'
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def run(*configs, group = None):
    config = configuration.load(*configs)
    if group:
        config.group = config.group + "-" + str(group)
    if config.from_scratch:
        config.group = 'scratch-' + config.group
        config.name = 'scratch-' + config.name
    if config.log:
        wandb.init(project = 'explainable-asag',
                   group = config.group,
                   name = config.name,
                   config = config)
        config = wandb.config

    model = transformers.AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels = config.num_labels)

    if config.from_scratch:
        model.init_weights()

    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()

    train_dataloader = dataset.dataloader(
        val_mode = False,
        data_file = config.train_data,
        data_source = config.data_source,
        vocab_file = config.model_name,
        num_labels = config.num_labels,
        train_percent = config.train_percent,
        batch_size = config.batch_size,
        drop_last = config.drop_last,
        num_workers = config.num_workers)
    val_dataloader = dataset.dataloader(
        val_mode = True,
        data_file = config.val_data,
        data_source = config.data_source,
        vocab_file = config.model_name,
        num_labels = config.num_labels,
        train_percent = config.val_percent,
        batch_size = config.batch_size,
        drop_last = config.drop_last,
        num_workers = config.num_workers)

    optimizer = torch.optim.__dict__[config.optimizer](model.parameters(), lr = config.learn_rate, **config.optimizer_kwargs)

    # Hack to get any scheduler we want. transformers.get_scheduler does not implement e.g. linear_with_warmup.
    get_scheduler = {
        'linear_with_warmup': transformers.get_linear_schedule_with_warmup,
        'cosine_with_warmup': transformers.get_cosine_schedule_with_warmup,
        'constant_with_warmup': transformers.get_constant_schedule_with_warmup,
        'cosine_with_hard_restarts_with_warmup': transformers.get_cosine_with_hard_restarts_schedule_with_warmup}
    lr_scheduler = get_scheduler[config.scheduler](optimizer, *config.scheduler_args, **config.scheduler_kwargs)

    best_f1 = 0.0
    patience = 0
    epoch = 0
    log_line = ''
    try:
        #while lr_scheduler.last_epoch <= total_steps:
        while epoch < config.max_epochs:
            epoch += 1
            av_epoch_loss =  training.train_epoch(train_dataloader, model, optimizer, lr_scheduler, config.num_labels, cuda, log = config.log)
            #tidy stuff up every epoch
            gc.collect()
            torch.cuda.empty_cache()

            p,r,f1,val_acc = validation.val_loop(model, val_dataloader, cuda)
            log_line = f'model: {config.model_name} | epoch: {epoch} | av_epoch_loss {av_epoch_loss:.5f} | f1: {f1:.5f} | accuracy: {val_acc:.5f} \n'
            print(log_line[:-1])
            if config.log:
                wandb.log({'precision': p , 'recall': r , 'f1': f1 ,  'accuracy': val_acc,'av_epoch_loss': av_epoch_loss})
            if f1 > best_f1:
                if config.log:
                    this_model =  os.path.join(wandb.run.dir,config.name + '-best_f1.pt')
                    print("saving to: ", this_model)
                    torch.save([model.state_dict(), config.__dict__], this_model)
                    wandb.save('*.pt')
                best_f1 = f1
                patience = 0 #max((0, patience-1))
            elif config.max_patience:
                patience +=1
                if patience >= config.max_patience:
                    break
        # Move stuff off the gpu
        model.cpu()
        #This is for sure a kinda dumb way of doing it, but the least mentally taxing right now
        #optimizer = torch.optim.__dict__[config.optimizer](model.parameters(), lr = config.learn_rate)
        #Is this smarter?
        optimizer_to_cpu(optimizer)
        gc.collect()
        torch.cuda.empty_cache()
        #return model   #Gives Error

    except KeyboardInterrupt:
        if config.log:
            wandb.save('*.pt')
        #Move stuff off the gpu
        model.cpu()
        optimizer_to_cpu(optimizer)
        gc.collect()
        torch.cuda.empty_cache()
        #return model    #Gives Error


if __name__ == '__main__':
    fire.Fire(run)
