
def mean_average_precision(predict_df, ground_df):  
    MAP = 0
    for query_id, doc_list_str in predict_df.iterrows():
        doc_list = doc_list_str["doc"].split()
        ans_doc_set = set(ground_df.loc[query_id, "doc"].split())
        AP = 0
        rel_cnt = 0
        for i, doc in enumerate(doc_list):
            if doc in ans_doc_set:
                rel_cnt += 1
                AP += rel_cnt / (i + 1)
        print(f'predict:{rel_cnt}, ground:{len(ans_doc_set)}')
        AP /= min(len(ans_doc_set), 50)
        MAP += AP
    MAP /= len(predict_df)
    return MAP