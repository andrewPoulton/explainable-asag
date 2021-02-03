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
    log = True
    if log:
        wandb.init(project = 'explainable-asag', group = 'test-1-epoch' , name = experiment)
        #log_model_dir = wandb.run.dir
        log_model_dir = os.path.join('models', experiment)
        if not os.path.exists(log_model_dir):
            os.makedirs(log_model_dir)

    config = configs.load(experiment)
    batch_size = 8
    warmup_steps = 1024
    learn_rate = 1e-5
    train_percent = 100
    val_percent = 100
    num_workers = 4
    total_steps = 10000
    num_labels = 2
    max_epochs = 24

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
    num_epochs = 0
    try:
        #while lr_scheduler.last_epoch <= total_steps:
        while num_epochs < max_epochs:
            num_epochs += 1
            av_epoch_loss =  training.train_epoch(train_dataloader,model, optimizer, lr_scheduler, num_labels, total_steps, cuda, log = log)
            #tidy stuff up every epoch
            gc.collect()
            torch.cuda.empty_cache()

            p,r,f1,val_acc = validation.val_loop(model, val_dataloader, cuda)
            log_line = f'precision: {p:.5f} | recall: {r:.5f} | f1: {f1:.5f} | accuracy: {val_acc:.5f}\n'
            print(log_line[:-1])
            print('av_epoch_loss', av_epoch_loss)
            if log and f1 > best_f1:
                print("saving to: ", os.path.join(log_model_dir, f'full_bert_model_best_acc.pt'))
                torch.save([model.state_dict(), config.__dict__], os.path.join(log_model_dir, f'full_bert_model_best_f1.pt'))
                wandb.save('*.pt')
                best_f1 = f1
                patience = max((0, patience-1))
            elif log:
                patience +=1
                if patience >= 3:
                    break
            if av_epoch_loss < .2:
                break
        if log:
            torch.save([model.state_dict(), config.__dict__], os.path.join(log_model_dir, f'full_bert_model_{lr_scheduler.last_epoch}_steps.pt'))
        # Move stuff off the gpu
        model.cpu()
        #This is for sure a kinda dumb way of doing it, but the least mentally taxing right now
        optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)
        gc.collect()
        torch.cuda.empty_cache()
        return None        # returning model gives error
        #return model

    except KeyboardInterrupt:
        # if log:
        #     wandb.save('*.pt')
        #Move stuff off the gpu
        model.cpu()
        #This is for sure a kinda dumb way of doing it, but the least mentally taxing right now
        optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)
        gc.collect()
        torch.cuda.empty_cache()
        #return model


if __name__ == '__main__':
    fire.Fire(run)
