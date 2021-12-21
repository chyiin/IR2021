import json
import string


class QueryDataset():
    def __init__(self, path, mode):
        self.query_list = []

        self._parse_all_queries(path, mode)
        # print(self.query_list)

    def _parse_all_queries(self, path, mode):
        with open(path) as file:
            query_list = json.load(file)[f"{mode}_query"]
        # for key, value in query_list[0].items(): print(key, value)
        for query in query_list:
            term_list = []
            for text_type in ['note', 'description', 'summary']:
                text = query[text_type]
                cand_term_list = text.split()
                for term in cand_term_list:
                    term = term.strip(string.punctuation)
                    if term != "":
                        term_list.append(term)
            self.query_list.append({
                'id': query['index'],
                'term_list': term_list
            })

    def __len__(self):
        return len(self.query_list)

    def __getitem__(self, idx):
        return self.query_list[idx]


if __name__ == '__main__': # For dev
    q = QueryDataset('dataset/train_query.txt', 'train')
