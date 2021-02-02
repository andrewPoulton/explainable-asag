import dataset
import training
import configs
import transformers
import fire
import torch
#import utils



def run():
   failing = [
      "albert-base-squad2",
### TODO: Throws ERROR on loading tokenizer:
### ValueError: Couldn't instantiate the backend tokenizer from one of:
### (1) a `tokenizers` library serialization file,
### (2) a slow tokenizer instance to convert or (3) an equivalent slow tokenizer class to instantiate and convert.
### You need to have sentencepiece installed to convert a slow tokenizer to a fast one.
      "roberta-large-stsb",
      "distilroberta-base-stsb"
### TODO: Throw ERROR on computing loss in training. Something to do with non-matching batch sizes.
### with batch_size = 4 I got
### ValueError: Expected input batch_size (1) to match target batch_size (2).
### with batch_size = 1 I got
### RuntimeError: shape '[-1, 2]' is invalid for input of size 1
   ]
   for  experiment in [e for e in  configs.list_experiments() if e not in failing]:

        train_percent = 1 # for testing the training cycle locally
        total_steps = 64
        config = configs.load(experiment)
        loader = dataset.dataloader(
               data_file = 'data/flat_semeval5way_train.csv',
               data_source = "scientsbank",
               vocab_file = config.model_path,
               num_labels = config.num_labels,
               train_percent = train_percent,
               val_mode = False,
               random = True,
               batch_size = config.batch_size,
               drop_last = False,
               num_workers = 0) #num_workers = 4 gives error on my mac, cannot pickle local function

        model = transformers.AutoModelForSequenceClassification.from_pretrained(config.model_path)
        optimizer = torch.optim.Adam(model.parameters(), lr = config.learn_rate)
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)
        cuda = torch.cuda.is_available()
        num_labels = config.num_labels
        training.train_epoch(loader, model, optimizer, lr_scheduler, num_labels, total_steps, cuda)


if __name__ == '__main__':
    fire.Fire(run)
