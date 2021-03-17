import dataset
import training
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

def update_token_type_embeddings(module, embedding_size, initializer_range, num_token_types = 4):
    for attr_str in dir(module):
        if attr_str == "token_type_embeddings":
            old_embeddings = module.__getattr__(attr_str)
            new_embeddings = torch.nn.Embedding(num_token_types, embedding_size)
            new_embeddings.weight.data.normal_(mean = 0.0, std = initializer_range)
            setattr(module, attr_str, new_embeddings)
            print(f"Updated token_type_embedding from {old_embeddings} to {new_embeddings}.")
            return

    for n, ch in module.named_children():
        embeds = update_token_type_embeddings(ch, embedding_size, initializer_range, num_token_types)
        


def run(*configs, group = None):
    config = configuration.load(*configs)
    if config.group:
        config.group = config.data_source + '-' +  config.group
    else:
        config.group = config.data_source
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
    if config.token_types:
        embedding_size = model.config.__dict__.get('embedding_size', model.config.hidden_size)
        update_token_type_embeddings(model, embedding_size, model.config.initializer_range)
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
            av_epoch_loss =  training.train_epoch(train_dataloader, model, optimizer, lr_scheduler, config.num_labels, cuda, log = config.log, token_types = config.token_types)
            #tidy stuff up every epoch
            gc.collect()
            torch.cuda.empty_cache()

            p,r,f1,val_acc = training.val_loop(model, val_dataloader, cuda)
            log_line = f'model: {config.model_name} | epoch: {epoch} | av_epoch_loss {av_epoch_loss:.5f} | f1: {f1:.5f} | accuracy: {val_acc:.5f} \n'
            print(log_line[:-1])
            if config.log:
                wandb.log({'precision': p , 'recall': r , 'f1': f1 ,  'accuracy': val_acc,'av_epoch_loss': av_epoch_loss})
            if f1 > best_f1:
                if config.log:
                    this_model =  os.path.join(wandb.run.dir,'best_f1.pt')
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
        optimizer = torch.optim.__dict__[config.optimizer](model.parameters(), lr = config.learn_rate)
        gc.collect()
        torch.cuda.empty_cache()
        #return model   #Gives Error

    except KeyboardInterrupt:
        if config.log:
            wandb.save('*.pt')
        #Move stuff off the gpu
        model.cpu()
        optimizer = torch.optim.__dict__[config.optimizer](model.parameters(), lr = config.learn_rate)
        gc.collect()
        torch.cuda.empty_cache()
        #return model    #Gives Error


if __name__ == '__main__':
    fire.Fire(run)
