import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import string
import random
import numpy as np
import gensim.downloader as api

def compute_MAP(prediction_file: str, qrels_file: str) -> float:
    qrels = json.load(open(qrels_file))
    prediction = json.load(open(prediction_file))
    MAP = 0
    for topic in prediction:
        hit_num = 0
        for document in prediction[topic]:
            if qrels[topic]['document'][document] == 1:
                hit_num += 1
        doc_num = len(prediction[topic])
        prev_precision = 0
        now_hit_num = hit_num
        for index, document in enumerate(prediction[topic][::-1]):
            if qrels[topic]['document'][document] == 1:
                prev_precision = max(prev_precision, now_hit_num / (doc_num - index))
                now_hit_num -= 1
                MAP += prev_precision
        MAP /= qrels[topic]['relevant']
    MAP /= len(prediction)
    return MAP

class DRMMDataset(Dataset):
    def __init__(self, qrels_file, topics_file, docs_dir, use_topic=["title", "description"]):
        with open(qrels_file) as f_qrel:
            self.qrels = json.load(f_qrel)
        self.pos_docs = dict()
        self.neg_docs = dict()
        with open(topics_file) as f_topic:
            self.topics = json.load(f_topic)
        self.use_topic = use_topic
        self.docs_dir = Path(docs_dir)
        print('Loading word2vec model...')
        self.wv = api.load('word2vec-google-news-300')

        for qid in self.qrels:
            self.pos_docs[qid] = list()
            self.neg_docs[qid] = list()
            for doc in self.qrels[qid]['document']:
                if self.qrels[qid]['document'][doc] == 1:
                    self.pos_docs[qid].append(doc)
                else:
                    self.neg_docs[qid].append(doc)
    def __len__(self):
        return len(self.topics)
    def __getitem__(self, index):
        qid = str(index + 301)
        query = ""
        for tag in self.use_topic:
            query += self.topics[qid][tag]
            query += ' '
        query = self.convertSentence(query)

        pos_doc = random.choice(self.pos_docs[qid])
        neg_doc = random.choice(self.neg_docs[qid])

        with open(self.docs_dir / pos_doc) as f_pos_doc:
            pos_doc_content = self.convertSentence(f_pos_doc.read())
        with open(self.docs_dir / neg_doc) as f_neg_doc:
            neg_doc_content = self.convertSentence(f_neg_doc.read())
        
        return torch.tensor(query), torch.tensor(pos_doc_content), torch.tensor(neg_doc_content)
    def convertSentence(self, s):
        vec = list()
        for word in s.split():
            # remove puctuation
            word = word.strip().translate(str.maketrans('', '', string.punctuation))
            if word == '':
                continue
            try:
                vec.append(self.wv[word])
            except:
                vec.append(np.zeros(self.wv.vector_size))
        return np.stack(vec)



if __name__ == '__main__':
    import argparse
    # parser = argparse.ArgumentParser(description='Prediction and Qrel files')
    # parser.add_argument('prediction_file', type=str, help="Prediction file in json format")
    # parser.add_argument('qrels_file', type=str, help="Qrel file in json format ")

    # argvs = parser.parse_args()
    # score = compute_MAP(argvs.prediction_file, argvs.qrels_file)
    # print(score)
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('qrels_file', type=str, help="Qrel file in json format")
    parser.add_argument('topics_file', type=str, help="Topic file in json format")
    parser.add_argument('docs_dir', type=str, help="Doc dir in json format")
    argvs = parser.parse_args()
    test = DRMMDataset(argvs.qrels_file, argvs.topics_file, argvs.docs_dir)
    print('Load done')
    a, b, c = test[0]
    print(a[0], a.shape, b.shape, c.shape)
