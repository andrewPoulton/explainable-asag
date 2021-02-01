import dataset
import training
import model_configs
import utils
import transformers
from transformers import AutoModelForSequenceClassification
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DataLoader
from types import SimpleNamespace
import fire
import torch

def run():
    ## init loader
    training_data = dataset.SemEvalDataset('data/flat_semeval5way_train.csv', "bert-base-uncased")
    sampler = RandomSampler(training_data)
    batch_sampler = BatchSampler(sampler, batch_size = 32, drop_last=False)
    loader = DataLoader(training_data, batch_sampler=batch_sampler, collate_fn=training_data.collater)

    ## init model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

    ## other parameters
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 1024, 1024*24)
    cuda  = False

    ## mock config
    config = SimpleNamespace(**{'num_labels':2, 'total_steps': 24*1024})
    #training.train_epoch(loader, model, optimizer, lr_scheduler, config, cuda)

    ## train_epoch
    for i, batch in enumerate(loader):
        pass
    # loss_fn = torch.nn.CrossEntropyLoss()
    # with tqdm(total=len(loader.batch_sampler)) as pbar:
    #     epoch_loss = 0.
    #     for i, batch in enumerate(loader):
    #         if cuda:
    #             batch.cuda()
    #         optimizer.zero_grad()
    #         mask = batch.generate_mask()
    #         logits = model(input_ids = batch.input, attention_mask = mask)
    #         logits = logits[0]
    #         # import pdb; pdb.set_trace()
    #         loss = loss_fn(logits.view(-1, config.num_labels), batch.labels.view(-1))
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    #         optimizer.step()
    #         lr_scheduler.step()


    #         if batch.labels.size(0)>1:
    #             acc = accuracy_score(batch.labels.cpu(), logits.cpu().detach().argmax(-1).squeeze())
    #         else:
    #             acc = 0.
    #         # if torch._np.isnan(loss.item()):
    #         # import pdb; pdb.set_trace()
    #         epoch_loss += loss.item()
    #         # if i % config.log_interval == 0:
    #         # wandb.log({"Train Accuracy": acc, "Train Loss": loss.item(), "Gradient Norm": grad_norm(model).item(), "Learning Rate": optimizer.param_groups[0]['lr']})
    #         pbar.set_description(f'global_step: {lr_scheduler.last_epoch}| loss: {loss.item():.4f}| acc: {acc*100:.1f}%| epoch_av_loss: {epoch_loss/(i+1):.4f} |')
    #         pbar.update(1)
    #         if lr_scheduler.last_epoch > config.total_steps:
    #             break
    #     #  move stuff off GPU
    #     batch.cpu()
    #     logits = logits.cpu().detach().argmax(-1).squeeze()
    #     return epoch_loss/(i+1)


if __name__ == '__main__':
    fire.Fire(run)
