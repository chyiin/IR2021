import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer
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

# tokenize sentence function
def tokenize(text, MAX_LENGTH, tokenizer):

    input_ids = []
    attention_masks = []

    for sent in tqdm(text):

        encoded_dict = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True, 
                            max_length = MAX_LENGTH,           
                            pad_to_max_length = True,
                            return_attention_mask = True, 
                            return_tensors = 'pt',  
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    print(input_ids)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    print(input_ids.size())

    return input_ids, attention_masks

# load data
train_ans = pd.read_csv('data_clean/train_ans.csv')
document = pd.read_csv('data_clean/document.csv')
train_query = pd.read_csv('data_clean/train_query.csv')

tr_document = []
tr_query = []
tr_label = []
for i in range(len(train_ans['topic'][0:1])):
    tr_query.extend([train_query['query'][i]]*len(train_ans['doc'][i].split()))
    tr_label.extend([1]*len(train_ans['doc'][i].split()))
    pos_doc = document[document['doc'].isin(list(map(int, train_ans['doc'][i].split())))]['documen'].values
    tr_document.extend(pos_doc)

    tr_query.extend([train_query['query'][i]]*1000)
    tr_label.extend([0]*1000)
    neg_doc = document[document['doc'].isin(list(map(int, train_ans['doc'][i].split()))) == False].sample(n=1000)['documen'].values
    tr_document.extend(neg_doc)

training_data = pd.DataFrame({'query':tr_query, 'document':tr_document, 'label':tr_label})
training_data = training_data.sample(frac=1).reset_index(drop=True) # 15720

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

train_doc_input_ids, train_doc_attention_masks = tokenize(training_data['document'], 512, tokenizer)
train_que_input_ids, train_que_attention_masks = tokenize(training_data['query'], 512, tokenizer)

df = [train_doc_input_ids, train_doc_attention_masks, train_que_input_ids, train_que_attention_masks, list(map(int, training_data['label'].values))]

# with open(f'training_data.txt', 'w') as outfile:
#     json.dump(df, outfile)