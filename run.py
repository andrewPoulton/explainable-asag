import dataset
import training
import configs
import transformers
import fire
import torch
import wandb
#import utils

def run():
    experiments =[
      'bert-base',
      'bert-large',
      'roberta-base',
      'roberta-large',
      'albert-base',
       'albert-large',
      'distilbert-base-uncased',
      'distilroberta',
      'distilbert-base-squad2',
      'roberta-base-squad2',
      'distilroberta-base-squad2',
      'bert-base-squad2',
      'albert-base-squad2',
      ######## "roberta-large-stsb"
      ######## "distilroberta-base-stsb"
    ]
    for  experiment in  experiments:
        log = True
        if log:
            wandb.init(project = 'explainable-asag', name = experiment +'-1')
        config = configs.load(experiment)
        # mode to configs when decided on values
        batch_size = 8
        warmup_steps = 64
        learn_rate = 1e-5
        train_percent = 100
        total_steps = 10000

        loader = dataset.dataloader(
            data_file = 'data/flat_semeval5way_train.csv',
            data_source = "scientsbank",
            vocab_file = config.model_path,
            num_labels = config.num_labels,
            train_percent = train_percent,
            val_mode = False,
            random = True,
            batch_size = batch_size,
            drop_last = False,
            num_workers = 4)

        model = transformers.AutoModelForSequenceClassification.from_pretrained(config.model_path)
        optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        num_labels = config.num_labels
        cuda = torch.cuda.is_available()
        if cuda:
            model.cuda()

        training.train_epoch(loader, model, optimizer, lr_scheduler, num_labels, total_steps, cuda, log = log)


if __name__ == '__main__':
    fire.Fire(run)
