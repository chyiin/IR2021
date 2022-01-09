Information Retrieval and Extraction Final Project
===

# Preprocessing
```
python3 preprocessing.py
```

# Approach 1: Tfidf + BM25 + SentenceTransformer
```
# Ranking on Training Data
python3 TFIDF+BM25+SentenceTransformer.py --query data_clean/train_query.csv

# Ranking on Testing Data
python3 TFIDF+BM25+SentenceTransformer.py --query data_clean/test_query.csv
```

# Approach 2: Tfidf + BM25 + BertModel
```
# Train BertModel
python3 train_bert.py

# Ranking on Training Data
python3 TFIDF+BM25+BERT.py --query data_clean/train_query.csv

# Ranking on Testing Data
python3 TFIDF+BM25+BERT.py --query data_clean/test_query.csv
```

# Approach 3: Pyterrier (BM25 + PL2 + DPH)
```
# run in pyterrier.ipynb
```