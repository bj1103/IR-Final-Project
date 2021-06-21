import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import string
import random
import numpy as np
import gensim.downloader as api
from rank_bm25 import BM25Okapi



class DRMMDataset(Dataset):
    def __init__(self, qrels_file, topics_file, docs_file, word_model=None, mode='train', use_tag=["title", "description"]):
        self.pos_docs = dict()
        self.neg_docs = dict()
        self.mode = mode
        self.use_tag = use_tag
        self.translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        if word_model is None:
            word_model = api.load('glove-twitter-25')
        self.word2id = word_model.key_to_index

        with open(qrels_file) as f_qrel:
            self.qrels = json.load(f_qrel)
        with open(topics_file) as f_topic:
            self.topics = json.load(f_topic)
        with open(docs_file) as f_docs:
            self.docs = json.load(f_docs)

        # Use bm25 to compute idf
        print('Computing IDF...', end='')
        corpus = list()
        for doc in self.docs:
            corpus_tokens = self.docs[doc].translate(self.translator).strip().lower().split()
            corpus.append([token.strip() for token in corpus_tokens if token.strip() != ''])
        self.idf = BM25Okapi(corpus).idf
        print('Done')
        del corpus, corpus_tokens

        total_qids = list(self.qrels.keys())
        total_qids = np.array([int(qid) for qid in total_qids])
        if self.mode == 'train':
            indexs = list(set(range(len(total_qids))) - set(range(0, len(total_qids), 5)))
        else:
            indexs = list(range(0, len(total_qids), 5))

        self.qids = total_qids[indexs]

        for qid in self.qids:
            qid = str(qid)
            self.pos_docs[qid] = list()
            self.neg_docs[qid] = list()
            for doc in self.qrels[qid]['document']:
                if self.qrels[qid]['document'][doc] > 0:
                    self.pos_docs[qid].append(doc)
                else:
                    self.neg_docs[qid].append(doc)

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, index):
        qid = str(self.qids[index])
        query = ''
        for tag in self.use_tag:
            query += self.topics[qid][tag]
            query += ' '
        query, q_idf = self.convert_sentence(query.lower())

        pos_doc = random.choice(self.pos_docs[qid])
        neg_doc = random.choice(self.neg_docs[qid])

        pos_doc_content, _ = self.convert_sentence(self.docs[pos_doc].lower())
        neg_doc_content, _ = self.convert_sentence(self.docs[neg_doc].lower())
        
        return query, pos_doc_content, neg_doc_content, q_idf

    def convert_sentence(self, s):
        vec = list()
        w_idf = list()
        for word in s.split():
            # remove puctuation
            word = word.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).strip().split()
            for w in word:
                try:
                    vec.append(self.word2id[w])
                    w_idf.append(self.idf[w])
                except:
                    continue
        return torch.tensor(vec), torch.tensor(w_idf)

def collate_batch(batch):
    q, p, n, idf = zip(*batch)
    batch_size = len(batch)
    l = torch.tensor([q_vec.shape[0] for q_vec in q])
    q = torch.reshape(torch.nn.utils.rnn.pad_sequence(q), (batch_size, -1))
    p = torch.reshape(torch.nn.utils.rnn.pad_sequence(p), (batch_size, -1))
    n = torch.reshape(torch.nn.utils.rnn.pad_sequence(n), (batch_size, -1))
    idf = torch.reshape(torch.nn.utils.rnn.pad_sequence(idf), (batch_size, -1))
    return q, p, n, l, idf

class rerankDataset(Dataset):
    def __init__(self, ranking_file, topics_file, docs_file, word_model=None, use_tag=["title", "description"]):
        self.use_tag = use_tag
        self.translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        if word_model is None:
            word_model = api.load('glove-twitter-25')
        self.word2id = word_model.key_to_index

        with open(ranking_file) as f_qrel:
            self.rank_list = json.load(f_qrel)
        with open(topics_file) as f_topic:
            self.topics = json.load(f_topic)
        with open(docs_file) as f_docs:
            self.docs = json.load(f_docs)
        
        self.data = list()
        for qid in self.rank_list:
            for id, doc in enumerate(self.rank_list[qid]):
                self.data.append((qid, doc, id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        qid, doc, id = self.data[index]
        query = ''
        for tag in self.use_tag:
            query += self.topics[qid][tag]
            query += ' '
        query = self.convert_sentence(query.lower())

        doc_content = self.convert_sentence(self.docs[doc].lower())
        
        return query, doc_content, qid, id

    def convert_sentence(self, s):
        vec = list()
        for word in s.split():
            # remove puctuation
            word = word.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).strip().split()
            for w in word:
                try:
                    vec.append(self.word2id[w])
                except:
                    continue
        return torch.tensor(vec)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('qrels_file', type=str, help="Qrel file in json format")
    parser.add_argument('topics_file', type=str, help="Topic file in json format")
    parser.add_argument('docs_file', type=str, help="Clean documents in json format")
    argvs = parser.parse_args()
    test = DRMMDataset(argvs.qrels_file, argvs.topics_file, argvs.docs_file)
    print('Load data done')
    loader = DataLoader(test, batch_size=2, shuffle=False, collate_fn=collate_batch)
    for q, p, n, l, idf in loader:
        print(l, idf, sep='\n')
        input()