import dataset
import training
import validation
import configs
import transformers
import fire
import torch
import wandb
import gc
import os
#import utils

# experiments =[
#     'bert-base',
#     'bert-large',
#     'roberta-base',
#     'roberta-large',
#     'albert-base',
#     'albert-large',
#     'distilbert-base-uncased',
#     'distilroberta',
#     'distilbert-base-squad2',
#     'roberta-base-squad2',
#     'distilroberta-base-squad2',
#     'bert-base-squad2',
#     'albert-base-squad2',
#   ######## "roberta-large-stsb"
#   ######## "distilroberta-base-stsb"
# ]`

def run(experiment):
    config = configs.load(experiment)
    batch_size = config.batch_size
    warmup_steps = config.warmup_steps
    learn_rate = config.learn_rate
    total_steps = config.total_steps
    max_epochs = config.max_epochs
    num_labels = 2
    train_percent = 100
    val_percent = 100
    num_workers = 4
    log = True
    
    if log:
        wandb.init(project = 'explainable-asag',
                   group = 'full-run' , name = experiment,
                   config = config)

    #TODO: Fix to work with num_labels > 2
    model = transformers.AutoModelForSequenceClassification.from_pretrained(config.model_path) 
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()

    train_dataloader = dataset.dataloader(
        data_file = 'data/flat_semeval5way_train.csv',
        data_source = "scientsbank",
        vocab_file = config.model_path,
        num_labels = num_labels,
        train_percent = train_percent,
        val_mode = False,
        random = True,
        batch_size = batch_size,
        drop_last = False,
        num_workers = num_workers)
    val_dataloader = dataset.dataloader(
        data_file = 'data/flat_semeval5way_test.csv',
        data_source = "scientsbank",
        vocab_file = config.model_path,
        num_labels = num_labels,
        train_percent = val_percent,
        val_mode = True,
        random = True,
        batch_size = batch_size,
        drop_last = False,
        num_workers = num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = 0.0
    patience = 0
    epoch = 0
    try:
        #while lr_scheduler.last_epoch <= total_steps:
        while epoch < max_epochs:
            epoch += 1
            print('experiment:', experiment, 'epoch:', epoch)
            av_epoch_loss =  training.train_epoch(train_dataloader,model, optimizer, lr_scheduler, num_labels, total_steps, cuda, log = log)
            #tidy stuff up every epoch
            gc.collect()
            torch.cuda.empty_cache()

            p,r,f1,val_acc = validation.val_loop(model, val_dataloader, cuda)
            if log:
                wandb.log({'precision': p , 'recall': r , 'f1': f1 ,  'accuracy': val_acc,'av_epoch_loss': av_epoch_loss})
            log_line = f'epoch : {epoch} | precision: {p:.5f} | recall: {r:.5f} | f1: {f1:.5f} | accuracy: {val_acc:.5f}\n'
            print(log_line[:-1])
            print('av_epoch_loss', av_epoch_loss)
            if f1 > best_f1:
                if log:
                    model_path =  os.path.join(wandb.run.dir, experiment + '-best_f1.pt')
                    print("saving to: ", model_path)
                    torch.save([model.state_dict(), config.__dict__], model_path)
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
        optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)
        gc.collect()
        torch.cuda.empty_cache()
        return None        # returning model gives error
        #return model

    except KeyboardInterrupt:
        if log:
            wandb.save('*.pt')
        #Move stuff off the gpu
        model.cpu()
        #This is for sure a kinda dumb way of doing it, but the least mentally taxing right now
        optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)
        gc.collect()
        torch.cuda.empty_cache()
        #return model


if __name__ == '__main__':
    fire.Fire(run)
