import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import string
import random
import numpy as np
import gensim.downloader as api


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
        self.word2id = self.wv.key_to_index

        for qid in self.qrels:
            self.pos_docs[qid] = list()
            self.neg_docs[qid] = list()
            for doc in self.qrels[qid]['document']:
                if self.qrels[qid]['document'][doc] > 0:
                    self.pos_docs[qid].append(doc)
                else:
                    self.neg_docs[qid].append(doc)
    def __len__(self):
        return len(self.topics)
    def __getitem__(self, index):
        qid = str(index + 301)
        # print(f'=========={qid}===========')
        query = ""
        for tag in self.use_topic:
            query += self.topics[qid][tag]
            query += ' '
        query = self.convertSentence(query)

        pos_doc = random.choice(self.pos_docs[qid])
        neg_doc = random.choice(self.neg_docs[qid])

        with open(self.docs_dir / pos_doc) as f_pos_doc:
            # print(f'=============={pos_doc}==============')
            pos_doc_content = self.convertSentence(f_pos_doc.read())
        with open(self.docs_dir / neg_doc) as f_neg_doc:
            # print(f'=============={neg_doc}==============')
            neg_doc_content = self.convertSentence(f_neg_doc.read())
        
        return query, pos_doc_content, neg_doc_content
    def convertSentence(self, s):
        vec = list()
        for word in s.split():
            # remove puctuation
            word = word.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).strip().split()
            for w in word:
                w = w.lower()
                try:
                    vec.append(self.word2id[w])
                except:
                    # print(f'{w} fail')
                    continue
        return torch.tensor(vec)

def collate_batch(batch):
    q, p, n = zip(*batch)
    return (torch.nn.utils.rnn.pad_sequence(q), torch.nn.utils.rnn.pad_sequence(p), torch.nn.utils.rnn.pad_sequence(n))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('qrels_file', type=str, help="Qrel file in json format")
    parser.add_argument('topics_file', type=str, help="Topic file in json format")
    parser.add_argument('docs_dir', type=str, help="Doc dir in json format")
    argvs = parser.parse_args()
    test = DRMMDataset(argvs.qrels_file, argvs.topics_file, argvs.docs_dir)
    # print('Load done')
    # a, b, c = test[0]
    # print(b)
    loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=collate_batch)
    for ba in loader:
        print(ba)
        break