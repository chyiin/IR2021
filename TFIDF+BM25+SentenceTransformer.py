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

my_stop_words = text.ENGLISH_STOP_WORDS.union(["book"])

def same_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
same_seeds(42)

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
    sent_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

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

    comb_result = []
    for i in range(len(tfidf_bm25)):
        r = []
        print(f'Train Query {i+1}')
        for j in tqdm(range(len(tfidf_bm25[i]))):
            doc_inputs = torch.mean(torch.tensor(sent_model.encode(chunking(300, document[document['doc'] == int(tfidf_bm25[i][j])]['documen'].values[0]))), 0).tolist()
            query_inputs = sent_model.encode(input_query['query'][i]).tolist()
            r.append(float(cosine_similarity([doc_inputs], [query_inputs])[0]))
        values, indices = torch.topk(torch.tensor(r), 50)
        res = []
        for indx in indices:
            res.append(tfidf_bm25[i][int(indx)])
        comb_result.append(' '.join(list(map(str, res))))
    output_result = pd.DataFrame({'topic':input_query['topic'], 'doc':comb_result})
    output_result.to_csv('output.csv', index=False)
    print()

    if 'train' in args['query']:
        print('MAP:', mean_average_precision(output_result, train_ans))

if __name__ == '__main__':

    Method1(args)