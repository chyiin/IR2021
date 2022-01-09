import xml.etree.ElementTree as ET
import json
import re
import glob
from tqdm import tqdm
import pandas as pd

doc_filename = glob.glob("doc/*")

doc_index = []
doc_text = []
for index in tqdm(doc_filename):
    tree = ET.parse(index)
    root = tree.getroot()
    content = ET.tostring(root, encoding='utf8').decode('utf8')
    pattern = re.compile ('<[^<]*?>')
    tmp = pattern.findall(content, re.DOTALL)
    big_regex = re.compile('|'.join(map(re.escape, tmp)))
    the_message = big_regex.sub(' ', content)
    doc_index.append(index[index.rfind('/')+1:])
    doc_text.append(' '.join(' '.join([line.strip() for line in the_message[the_message.rfind('>')+1:].strip().splitlines()]).split()))

documnet_clean = pd.DataFrame({'doc':doc_index, 'document':doc_text})
documnet_clean["doc"] = pd.to_numeric(documnet_clean["doc"])
document = documnet_clean.sort_values(by=['doc']).reset_index(drop=True)
documnet_clean.to_csv('data_clean/document.csv', index=False)
        
# train_query

train_query_filename = glob.glob("train_query/*")

train_topic = []
train_query = []
for index in train_query_filename:
    tree = ET.parse(index)
    root = tree.getroot()
    contents = []
    for child in root:
        contents.append(child.text)
    train_topic.append(index[index.rfind('/')+1:])
    train_query.append(' '.join(' '.join([line.strip() for line in contents[2].strip().splitlines()]).split()))

train = pd.DataFrame({'topic':train_topic, 'query':train_query})
train["topic"] = pd.to_numeric(train["topic"])
train = train.sort_values(by=['topic']).reset_index(drop=True)
train.to_csv('data_clean/train_query.csv', index=False)

# test_query

test_query_filename = glob.glob("test_query/*")

test_topic = []
test_query = []
for index in test_query_filename:
    tree = ET.parse(index)
    root = tree.getroot()
    contents = []
    for child in root:
        contents.append(child.text)
    test_topic.append(index[index.rfind('/')+1:])
    test_query.append(' '.join(' '.join([line.strip() for line in contents[2].strip().splitlines()]).split()))

test = pd.DataFrame({'topic':test_topic, 'query':test_query})
test["topic"] = pd.to_numeric(test["topic"])
test = test.sort_values(by=['topic']).reset_index(drop=True)
test.to_csv('data_clean/test_query.csv', index=False)