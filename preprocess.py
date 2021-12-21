
import xml.etree.ElementTree as ET
import json
import re
import glob
from tqdm import tqdm

doc_filename = glob.glob("doc/*")

document = {}
document['document'] = []
for index in tqdm(doc_filename):
    tree = ET.parse(index)
    root = tree.getroot()
    content = ET.tostring(root, encoding='utf8').decode('utf8')
    pattern = re.compile ('<[^<]*?>')
    tmp = pattern.findall(content, re.DOTALL)
    big_regex = re.compile('|'.join(map(re.escape, tmp)))
    the_message = big_regex.sub(' ', content)
    document['document'].append({
    'index': index[index.rfind('/')+1:],
    'content': the_message,
    })

with open('document.txt', 'w') as outfile:
    json.dump(document, outfile)
# with open('document.txt') as json_file:
#     tr_data = json.load(json_file)
#     for i in tr_data['document']:
#         print(i['index'])
#         print(i['content'])
#         input('1')
        
# train_query

train_query_filename = glob.glob("train_query/*")

tr_data = {}
tr_data['train_query'] = []
for index in train_query_filename:
    tree = ET.parse(index)
    root = tree.getroot()
    contents = []
    for child in root:
        contents.append(child.text)
    tr_data['train_query'].append({
    'index': index[index.rfind('/')+1:],
    'note': contents[0],
    'description': contents[1],
    'summary': contents[2],
    })

with open('train_query.txt', 'w') as outfile:
    json.dump(tr_data, outfile)
# with open('train_query.txt') as json_file:
#     tr_data = json.load(json_file)
#     for p in tr_data['train_query']:
#         print(p['index'])
#         print(p['note'])
#         print(p['description'])
#         print(p['summary'])

# test_query

test_query_filename = glob.glob("test_query/*")

test_data = {}
test_data['test_query'] = []
for index in test_query_filename:
    tree = ET.parse(index)
    root = tree.getroot()
    contents = []
    for child in root:
        contents.append(child.text)
    test_data['test_query'].append({
    'index': index[index.rfind('/')+1:],
    'note': contents[0],
    'description': contents[1],
    'summary': contents[2],
    })

with open('test_query.txt', 'w') as outfile:
    json.dump(test_data, outfile)
# with open('test_query.txt') as json_file:
#     data = json.load(json_file)
#     for p in data['train_query']:
#         print(p['index'])
#         print(p['note'])
#         print(p['description'])
#         print(p['summary'])
