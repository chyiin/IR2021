import argparse
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ans_path", type=Path, default="dataset/train_ans.csv")
    parser.add_argument("--res_path", type=Path, default="dataset/train_result.csv")
    args = parser.parse_args()

    ans_df = pd.read_csv(args.ans_path).set_index("topic")
    res_df = pd.read_csv(args.res_path).set_index("topic")

    MAP = 0
    for query_id, doc_list_str in res_df.iterrows():
        doc_list = doc_list_str["doc"].split()[:50]
        ans_doc_set = set(ans_df.loc[query_id, "doc"].split())
        AP = 0
        rel_cnt = 0
        for i, doc in enumerate(doc_list):
            if doc in ans_doc_set:
                rel_cnt += 1
                AP += rel_cnt / (i + 1)
        AP /= min(len(ans_doc_set), 50)
        print(AP)
        MAP += AP
    
    MAP /= len(res_df)
    print(MAP)
