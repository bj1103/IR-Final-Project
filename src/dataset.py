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
    def __init__(self, qrels_file, query_id_file, docs_id_file, idf_file, mode='train'):
        self.pos_docs = dict()
        self.neg_docs = dict()

        with open(qrels_file) as f_qrel:
            self.qrels = json.load(f_qrel)
        with open(query_id_file) as f_query_id:
            self.query_id = json.load(f_query_id)
        with open(docs_id_file) as f_docs_id:
            self.docs_id = json.load(f_docs_id)
        with open(idf_file) as f_idf:
            self.idf = json.load(f_idf)
        self.qids = get_qids(mode, self.qrels)
        self.doc_num = len(self.docs_id)

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
        query = self.query_id[qid]
        q_idf = [self.idf[str(wid)] if str(wid) in self.idf else np.log(self.doc_num) for wid in query]

        pos_doc = random.choice(self.pos_docs[qid])
        neg_doc = random.choice(self.neg_docs[qid])

        pos_doc_embed = self.docs_id[pos_doc]
        neg_doc_embed = self.docs_id[neg_doc]

        return torch.tensor(query), torch.tensor(pos_doc_embed), torch.tensor(neg_doc_embed), torch.tensor(q_idf)

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
    def __init__(self, ranking_file, qrels_file, query_id_file, docs_id_file, idf_file):
        with open(ranking_file) as f_rank:
            self.rank_list = json.load(f_rank)
        with open(query_id_file) as f_query_id:
            self.query_id = json.load(f_query_id)
        with open(docs_id_file) as f_docs_id:
            self.docs_id = json.load(f_docs_id)
        with open(qrels_file) as f_qrels:
            self.qrels = json.load(f_qrels)
        with open(idf_file) as f_idf:
            self.idf = json.load(f_idf)
        self.qids = get_qids('test', self.qrels)
        self.doc_num = len(self.docs_id)

        self.data = list()
        for qid in self.qids:
            for id, doc in enumerate(self.rank_list[qid]):
                self.data.append((qid, doc, id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        qid, doc, id = self.data[index]
        qid = str(self.qids[index])
        query = self.query_id[qid]
        q_idf = [self.idf[str(wid)] if str(wid) in self.idf else np.log(self.doc_num) for wid in query]

        doc_embed = self.docs_id[doc]

        return torch.tensor(query), torch.tensor(doc_embed), torch.tensor(q_idf), id

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('qrels_file', type=str, help="Qrel file in json format")
    parser.add_argument('query_id_file', type=str, help="Embedded query in json format")
    parser.add_argument('docs_id_file', type=str, help="Embedded documents in json format")
    parser.add_argument('idf_file', type=str, help="IDF among documents in json format")
    argvs = parser.parse_args()
    test = DRMMDataset(argvs.qrels_file, argvs.query_id_file, argvs.docs_id_file, argvs.idf_file)
    print('Load data done')
    loader = DataLoader(test, batch_size=2, shuffle=False, collate_fn=collate_batch)
    for q, p, n, ql, pl, nl, idf in loader:
        print(f'===query===\n{ql}\n{q}\n===pos_doc===\n{pl}\n{p}\n===neg_doc===\n{nl}\n{n}\n{idf}')
        input()
