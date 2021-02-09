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

def run(*configs, group = None):
    config = configuration.load(*configs)
    if group:
        config.group = 'group-' + str(group)
    if config.log:
        wandb.init(project = 'explainable-asag',
                   group = config.group + '-scratch' if config.from_scratch else config.group,
                   name = config.name + '-scratch' if config.from_scratch else config.name,
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

    optimizer = torch.optim.Adam(model.parameters(), lr = config.learn_rate)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, config.warmup_steps, config.total_steps)

    best_f1 = 0.0
    patience = 0
    epoch = 0
    try:
        #while lr_scheduler.last_epoch <= total_steps:
        while epoch < config.max_epochs:
            epoch += 1
            av_epoch_loss =  training.train_epoch(train_dataloader,model, optimizer, lr_scheduler, config.num_labels, config.total_steps, cuda, log = config.log)
            #tidy stuff up every epoch
            gc.collect()
            torch.cuda.empty_cache()

            p,r,f1,val_acc = validation.val_loop(model, val_dataloader, cuda)
            if config.log:
                wandb.log({'precision': p , 'recall': r , 'f1': f1 ,  'accuracy': val_acc,'av_epoch_loss': av_epoch_loss})
            log_line = f'epoch: {epoch} | precision: {p:.5f} | recall: {r:.5f} | f1: {f1:.5f} | accuracy: {val_acc:.5f}\n'
            print(log_line[:-1])
            print('av_epoch_loss', av_epoch_loss)
            if f1 > best_f1:
                if config.log:
                    this_model =  os.path.join(wandb.run.dir,config.name + '-best_f1.pt')
                    print("saving to: ", this_model_name)
                    torch.save([model.state_dict(), config.__dict__], this_model)
                    wandb.save('*.pt')
                best_f1 = f1
                patience = 0 #max((0, patience-1))
            else:
                patience +=1
                if patience >= 3:
                    break
        # Move stuff off the gpu
        model.cpu()
        #This is for sure a kinda dumb way of doing it, but the least mentally taxing right now
        optimizer = torch.optim.Adam(model.parameters(), lr = config.learn_rate)
        gc.collect()
        torch.cuda.empty_cache()
        #return model   #Gives Error, no iputs

    except KeyboardInterrupt:
        if config.log:
            wandb.save('*.pt')
        #Move stuff off the gpu
        model.cpu()
        #This is for sure a kinda dumb way of doing it, but the least mentally taxing right now
        optimizer = torch.optim.Adam(model.parameters(), lr = config.learn_rate)
        gc.collect()
        torch.cuda.empty_cache()
        #return model    #Gives Error, no iputs


if __name__ == '__main__':
    fire.Fire(run)
