import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from torch.utils.data import TensorDataset, random_split
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from model import Encoder1
from map import mean_average_precision
import argparse
import json

# set seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(3)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(42)

# device setting    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

def load(filename):

    tr_doc, tr_doc_mask, tr_que, tr_que_mask, tr_label = [], [], [], [], []
    with open(filename) as json_file:
        tr_data = json.load(json_file)
        for p1 in tr_data[0]:
            tr_doc.append(torch.tensor([p1]))
        for p2 in tr_data[1]:
            tr_doc_mask.append(torch.tensor([p2]))
        for p3 in tr_data[2]:
            tr_que.append(torch.tensor([p3]))
        for p4 in tr_data[3]:
            tr_que_mask.append(torch.tensor([p4]))
        for p5 in tr_data[4]:
            tr_label.append(torch.tensor([float(p5)]))

    return torch.cat(tr_doc, dim=0), torch.cat(tr_doc_mask, dim=0), torch.cat(tr_que, dim=0), torch.cat(tr_que_mask, dim=0), torch.tensor(tr_label)

def TrainBertModel():

    train_doc_input_ids, train_doc_attention_masks, train_que_input_ids, train_que_attention_masks, train_labels = load('training_data.txt')

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(train_doc_input_ids, train_doc_attention_masks, train_que_input_ids, train_que_attention_masks, train_labels)

    # Create a train-validation split.
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    # Divide the dataset by randomly selecting samples.
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    batch_size = 4

    # Create the DataLoaders for our training and validation sets.
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
    valid_dataloader = DataLoader(valid_dataset,sampler = SequentialSampler(valid_dataset),batch_size = batch_size)

    model = Encoder1(pretrained='bert-base-cased', hidden_size=768).cuda()

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer1 = list(model.named_parameters())
        no_decay1 = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters1 = [
            {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in no_decay1)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in no_decay1)],
            'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer1 = list(model.classifier.named_parameters())
        optimizer_grouped_parameters1 = [{"params": [p for n, p in param_optimizer1]}]
        
    optimizer = AdamW(optimizer_grouped_parameters1, lr=1e-5)

    epoch = 10
    max_grad_norm = 1.0
    total_steps = len(train_dataloader) * epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    eval_loss_list = []
    patient = 0
    for _ in range(epoch):

        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):

            batch = tuple(t.to(device) for t in batch)
            doc_input_ids, doc_input_mask, que_input_ids, que_input_mask, b_labels = batch
            model.zero_grad()

            outputs = model(input_ids=doc_input_ids, query_ids=que_input_ids, token_type_ids=None,
                            attention_mask=doc_input_mask, que_attention_mask=que_input_mask, labels=b_labels)
            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        model.eval()

        eval_loss = 0
        
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            doc_input_ids, doc_input_mask, que_input_ids, que_input_mask, b_labels = batch

            with torch.no_grad():

                outputs = model(input_ids=doc_input_ids, query_ids=que_input_ids, token_type_ids=None,
                    attention_mask=doc_input_mask, que_attention_mask=que_input_mask, labels=b_labels)

            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            eval_loss += outputs[0].item()
            
        eval_loss = eval_loss / len(valid_dataloader)    
        eval_loss_list.append(eval_loss)    
        print(f'''Epoch [{_+1}/{epoch}] total loss complete. Train Loss: {avg_train_loss:.5f}. Val Loss: {eval_loss:.5}''') 
        
        # condition setting (model saved)
        if eval_loss <= min(eval_loss_list):
            patient = 0
            print("saving state dict")
            torch.save(model.state_dict(), f"bert_model.pt")
        else:
            # early stopping
            patient += 1
            if patient == 3:
                print(f'Early Stop.')
                break
        print()     


if __name__ == '__main__':

    TrainBertModel()