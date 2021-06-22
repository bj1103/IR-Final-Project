import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import string
import random
import numpy as np
import gensim.downloader as api
from rank_bm25 import BM25Okapi
from utils import get_qids



class DRMMDataset(Dataset):
    def __init__(self, qrels_file, topics_file, docs_file, folds_file, word_model=None, mode='train', test_folds=[4], use_tag=["title", "description"]):
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
        self.qids = get_qids(folds_file, mode, test_folds, self.qrels)

        # Use bm25 to compute idf
        print('Computing IDF...', end='')
        corpus = list()
        for doc in self.docs:
            corpus_tokens = self.docs[doc].translate(self.translator).strip().lower().split()
            corpus.append([token.strip() for token in corpus_tokens if token.strip() != ''])
        self.idf = BM25Okapi(corpus).idf
        print('Done')
        del corpus, corpus_tokens

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
    # q_len = torch.tensor([q_vec.shape[0] for q_vec in q])
    # p_len = torch.tensor([p_vec.shape[0] for p_vec in p])
    # n_len = torch.tensor([n_vec.shape[0] for n_vec in n])
    q = torch.nn.utils.rnn.pad_sequence(q).T
    p = torch.nn.utils.rnn.pad_sequence(p).T
    n = torch.nn.utils.rnn.pad_sequence(n).T
    idf = torch.nn.utils.rnn.pad_sequence(idf).T
    return q, p, n, idf

class rerankDataset(Dataset):
    def __init__(self, ranking_file, topics_file, docs_file, qrels_file, folds_file, word_model=None, test_folds=[4],  use_tag=["title", "description"]):
        self.use_tag = use_tag
        self.translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        if word_model is None:
            word_model = api.load('glove-twitter-25')
        self.word2id = word_model.key_to_index

        with open(ranking_file) as f_rank:
            self.rank_list = json.load(f_rank)
        with open(topics_file) as f_topic:
            self.topics = json.load(f_topic)
        with open(docs_file) as f_docs:
            self.docs = json.load(f_docs)
        with open(qrels_file) as f_qrels:
            self.qrels = json.load(f_qrels)
        self.qids = get_qids(folds_file, 'test', test_folds, self.qrels)

        corpus = list()
        for doc in self.docs:
            corpus_tokens = self.docs[doc].translate(self.translator).strip().lower().split()
            corpus.append([token.strip() for token in corpus_tokens if token.strip() != ''])
        self.idf = BM25Okapi(corpus).idf
        print('Done')
        del corpus, corpus_tokens

        self.data = list()
        for qid in self.qids:
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
        query, q_idf = self.convert_sentence(query.lower())

        doc_content, _ = self.convert_sentence(self.docs[doc].lower())

        return query, doc_content, q_idf, qid, id

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
    for q, p, n, ql, pl, nl, idf in loader:
        print(f'===query===\n{ql}\n{q}\n===pos_doc===\n{pl}\n{p}\n===neg_doc===\n{nl}\n{n}\n{idf}')
        input()
