from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from tqdm import tqdm

def metrics(predictions, y_true, metric_params):
    precision = precision_score(y_true, predictions, **metric_params)
    recall = recall_score(y_true, predictions, **metric_params)
    f1 = f1_score(y_true, predictions, **metric_params)
    accuracy = accuracy_score(y_true, predictions)
    return precision, recall, f1, accuracy

@torch.no_grad()
def val_loop(model, loader, cuda):
    model.eval()
    # batches = list(loader)
    preds = []
    true_labels = []
    with tqdm(total= len(loader.batch_sampler)) as pbar:
        for i,batch in enumerate(loader):
            if cuda:
                batch.cuda()
            mask = batch.generate_mask()
            logits = model(input_ids = batch.input, attention_mask = mask)
            logits = logits[0]
            preds.append(logits.argmax(-1).squeeze().cpu())
            true_labels.append(batch.labels.cpu())
            pbar.update(1)
    preds = torch.cat(preds)
    y_true = torch.cat(true_labels)
    model.train()
    metric_params = {'average':'weighted', 'labels':list(range(model.config.num_labels))}
    return metrics(preds, y_true, metric_params)
