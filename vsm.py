from collections import defaultdict
from pathlib import Path
import numpy as np
import math
import pickle
from tqdm import tqdm
from doc_dataset import DocDataset
from query_dataset import QueryDataset
import csv

class VSM():
    def __init__(self, mode, model_dir, doc_dataset, vsm_dir=None):
        self.mode = mode
        self.model_dir = Path(model_dir)

        if self.mode == 'dev':
            self.vsm_dir = Path(vsm_dir)
            self.vsm_dir.mkdir(parents=True, exist_ok=True)

        self.doc_dataset = doc_dataset

        self.rep_doc_list = [{} for _ in range(self.doc_dataset.n_docs)]
        # self.rep_doc_list = [{} for _ in range(1000)]

        self.term_to_idf = {}

    def build_model(self):
        for term, term_data in tqdm(self.doc_dataset.inverted_file.items()):
            # if type(term) == int:
            #     vocab_id_1 = term
            #     vocab_id_2 = -1
            # else:
            #     vocab_id_1, vocab_id_2 = [int(s) for s in term.split("_")]
            df = term_data["df"]

            term_idf = self._calc_idf(df)

            for doc_data in term_data["list"]:
                doc_idx = doc_data["doc_idx"]
                self.rep_doc_list[doc_idx][self.doc_dataset.vocab[term]] = self._calc_tf_Okapi(
                    tf=doc_data["tf"],
                    # dl=self.doc_dataset.get_dl(doc_idx)
                    dl=self.doc_dataset.avdl
                ) * term_idf

            self.term_to_idf[self.doc_dataset.vocab[term]] = term_idf

        self.rep_doc_list = [self._normalize_rep(rep) for rep in self.rep_doc_list]

        if self.mode == 'dev':
            self._save()
    
    def calc_rep_query(self, query):
        if self.mode == 'dev':
            self._load()
        
        query_term_list = query['term_list']
        # print(query_term_list)
        dl = len(query_term_list)
        rep_query = defaultdict(int)

        for term in query_term_list:
            rep_query[term] += 1
            
        for term, tf in rep_query.items():
            idf = self.term_to_idf[term] if term in self.term_to_idf else 0
            rep_query[term] = self._calc_tf_Okapi(tf=tf, dl=self.doc_dataset.avdl) * idf
            # rep_query[term] = self._calc_tf_Okapi(tf=tf, dl=dl) * idf

        rep_query = self._normalize_rep(rep_query)

        return rep_query
        
    def search(self, rep_query):
        # print(rep_query, len(rep_query))
        sim_score_list = []

        for rep_doc in self.rep_doc_list:
            sim_score = 0
            for term, val in rep_query.items():
                if term in rep_doc:
                    # print(term, rep_doc[term])
                    sim_score += rep_doc[term] * val
            sim_score_list.append(sim_score)
            # print(sim_score)
        sim_score_list = np.array(sim_score_list)

        sim_score_list[np.isnan(sim_score_list)] = float('-inf')

        rank_list = sim_score_list.argsort()[::-1]

        # rank_list = rank_list[sim_score_list[rank_list] >= 0.2]
        # print(sim_score_list[rank_list][:50])

        return rank_list

    def relevance_feedback(self, rep_query, pos_doc_list, neg_doc_list, alpha=1, beta=0.75, gamma=0.15):
        new_rep = {}
        
        for term, val in rep_query.items():
            new_rep[term] = val * alpha
        
        for i in pos_doc_list:
            rep = self.rep_doc_list[i]
            for term, val in rep.items():
                if term in new_rep:
                    new_rep[term] += beta/len(pos_doc_list) * val
                else:
                    new_rep[term] = beta/len(pos_doc_list) * val
        
        for i in neg_doc_list:
            rep = self.rep_doc_list[i]
            for term, val in rep.items():
                if term in new_rep:
                    new_rep[term] -= gamma/len(neg_doc_list) * val
                else:
                    new_rep[term] = -gamma/len(neg_doc_list) * val

        return self._normalize_rep(new_rep)

    def _normalize_rep(self, rep):
        norm_of_rep = 0
        for val in rep.values():
            norm_of_rep += val**2
        norm_of_rep = math.sqrt(norm_of_rep)

        for term, val in rep.items():
            rep[term] = val / norm_of_rep

        return rep

    def _calc_tf_Okapi(self, tf, dl, k=2, b=0.75):
        avdl = self.doc_dataset.avdl
        return (k + 1)*tf / (k*((1 - b) + b*dl/avdl) + tf)

    def _calc_idf_Okapi(self, df):
        N = self.doc_dataset.n_docs
        return math.log((N - df + 0.5) / (df + 0.5))

    def _calc_idf(self, df):
        N = self.doc_dataset.n_docs
        return math.log((N + 1) / df)

    def _save(self):
        with open(self.vsm_dir / 'rep_doc_list.npy', 'wb') as file:
            pickle.dump(self.rep_doc_list, file)
        with open(self.vsm_dir / 'term_to_idf.npy', 'wb') as file:
            pickle.dump(self.term_to_idf, file)

    def _load(self):
        with open(self.vsm_dir / 'rep_doc_list.npy', 'rb') as file:
            self.rep_doc_list = pickle.load(file)
        with open(self.vsm_dir / 'term_to_idf.npy', 'rb') as file:
            self.term_to_idf = pickle.load(file)


if __name__ == '__main__': # For dev

    doc_dataset = DocDataset('dataset/document.txt')

    with open('model/doc_dataset.npy', 'wb') as file:
        pickle.dump(doc_dataset, file)
    with open('model/doc_dataset.npy', 'rb') as file:
        doc_dataset = pickle.load(file)

    vsm = VSM('dev', 'model/', doc_dataset, 'model/vsm')
    vsm.build_model()
    # exit()

    n_pos = 5
    n_neg = 0
    k = 50

    with open('dataset/train_result.csv', 'w') as file:
        csv_writer = csv.DictWriter(file, ['topic', 'doc'])
        csv_writer.writeheader()

        query_dataset = QueryDataset('dataset/train_query.txt', 'train')
        for query in tqdm(query_dataset):
            rep_query = vsm.calc_rep_query(query)
            rank_list = vsm.search(rep_query)

            rep_query = vsm.relevance_feedback(rep_query, rank_list[:n_pos], rank_list[n_pos:n_pos+n_neg])

            rank_list = vsm.search(rep_query)

            rank_list = rank_list[:k]

            csv_writer.writerow({'topic': query['id'], 'doc': ' '.join([str(vsm.doc_dataset.doc_id_list[idx]) for idx in rank_list])})
