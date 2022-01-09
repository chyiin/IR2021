import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from map import mean_average_precision
import argparse
from model import Encoder1
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

my_stop_words = text.ENGLISH_STOP_WORDS.union(["book"])

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

def tokenize(text, MAX_LENGTH, tokenizer, label=None):

    input_ids = []
    attention_masks = []

    for sent in tqdm(text):

        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = MAX_LENGTH,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    if label is not None:  
        labels = torch.tensor(list(map(int, label)))
        return input_ids, attention_masks, labels
    return input_ids, attention_masks


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, default='data_clean/train_query.csv', help='Dataset file name.')
    args = parser.parse_args()

    return args
    
args = parse_args()
args = vars(args)

def chunking(max_len, sent):
    tokenized_text = sent.lower().split(" ")
    # using list comprehension
    final = [tokenized_text[i * max_len:(i + 1) *max_len] for i in range((len(tokenized_text) + max_len - 1) // max_len)] 
    
    # join back to sentences for each of the chunks
    sent_chunk = []
    for item in final:
        sent_chunk.append(' '.join(item))
    return sent_chunk

def Method1(args):

    # load data
    train_ans = pd.read_csv('data_clean/train_ans.csv')
    document = pd.read_csv('data_clean/document.csv')
    input_query = pd.read_csv(args['query'])

    # tfidf
    tfidf = TfidfVectorizer(lowercase=True, stop_words=my_stop_words, min_df=4, ngram_range=(1,3))
    tfidf_X = tfidf.fit_transform(tqdm(document['documen']))

    # bm25
    tokenized_corpus = [doc.split(" ") for doc in document['documen']]
    bm25 = BM25Okapi(tqdm(tokenized_corpus))

    # reranked by SentenceTransformer through cosine similarity
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = Encoder1(pretrained='bert-base-cased', hidden_size=768).cuda()
    model.load_state_dict(torch.load('bert_model.pt')) 

    # ranking & filter data & select top50 relevant documents
    tfidf_result = []
    for i in tqdm(range(len(input_query['query']))):
        query = input_query['query'][i]
        query_vec = tfidf.transform([query])
        results = cosine_similarity(tfidf_X, query_vec)
        values, indices = torch.topk(torch.tensor(results).squeeze(-1), 500)
        r = []
        for idxx in indices:
            r.append(document['doc'][int(idxx)])
        tfidf_result.append(' '.join(list(map(str, r))))
    
    tfidf_df = pd.DataFrame({'topic':input_query['topic'], 'doc':tfidf_result})
    # tfidf_df.to_csv('filter_data_tfidf/tfidf_top50.csv', index=False)

    # ranking & filter & select top50 relevant documents
    bm_result = []
    for i in tqdm(range(len(input_query['query']))):
        query = input_query['query'][i]
        tokenized_query = query.split(" ")
        results = bm25.get_scores(tokenized_query)
        values, indices = torch.topk(torch.tensor(results).squeeze(-1), 500)
        r = []
        for idxx in indices:
            r.append(document['doc'][int(idxx)])
        bm_result.append(' '.join(list(map(str, r))))
    bm25_df = pd.DataFrame({'topic':input_query['topic'], 'doc':bm_result})
    # bm25_df.to_csv('filter_data_bm25/bm25_top50.csv', index=False)

    # tfidf + bm25 
    # filer overlap data
    tfidf_bm25 = []
    for i in range(len(tfidf_result)):
        tfidf_bm25.append(list(set(tfidf_result[i].split()+bm_result[i].split())))  

    test_topic = []
    test_doc = []
    test_query = []
    test_document = []
    for i in range(len(tfidf_bm25)):
        for j in range(len(tfidf_bm25[i])):
            test_query.append(input_query['query'][i])
            test_topic.append(input_query['topic'][i])
            test_document.append(document[document['doc'] == int(tfidf_bm25[i][j])]['documen'].values[0])
            test_doc.append(int(tfidf_bm25[i][j]))
    test_doc_input_ids, test_doc_attention_masks = tokenize(test_document, 512, tokenizer)
    test_que_input_ids, test_que_attention_masks = tokenize(test_query, 512, tokenizer)

    batch_size = 1  
    # Create the DataLoader.
    prediction_data = TensorDataset(test_doc_input_ids, test_doc_attention_masks, test_que_input_ids, test_que_attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size, shuffle=False)

    # prediction step
    model.eval()
    predictions , probability = [], []
    for batch in tqdm(prediction_dataloader):
        batch = tuple(t.to(device) for t in batch)
        doc_input_ids, doc_input_mask, que_input_ids, que_input_mask = batch

        with torch.no_grad():

            outputs = model(input_ids=doc_input_ids, query_ids=que_input_ids, token_type_ids=None,
                attention_mask=doc_input_mask, que_attention_mask=que_input_mask)

        predictions.append(float(outputs[1].detach().cpu()[0][0]))    

    output_result = pd.DataFrame({'topic':test_topic, 'doc':test_doc, 'prob':predictions})
    grouped_df = output_result.groupby("topic")
    grouped_doc = grouped_df["doc"].apply(list)
    grouped_doc = grouped_doc.reset_index()
    grouped_prob = grouped_df["prob"].apply(list)
    grouped_prob = grouped_prob.reset_index()

    final_output = []
    for i in range(len(grouped_doc['topic'])):
        values, indices = torch.topk(torch.tensor(grouped_prob['prob'][i]), 50)
        res = []
        for indx in indices:
            res.append(grouped_doc['doc'][i][int(indx)])
        final_output.append(' '.join(list(map(str, res))))

    final = pd.DataFrame({'topic':input_query['topic'], 'doc':final_output})
    final.to_csv('tfidf_bm_bert.csv', index=False)

    if 'train' in args['query']:
        print('MAP:', mean_average_precision(final, train_ans))

if __name__ == '__main__':

    Method1(args)