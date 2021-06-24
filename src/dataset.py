import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import numpy as np
import gensim
from utils import get_qids
import torch.nn as nn



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
        query = self.query_id[qid]
        q_idf = [self.idf[str(wid)] if str(wid) in self.idf else np.log(self.doc_num) for wid in query]

        doc_embed = self.docs_id[doc]

        return torch.tensor(query), torch.tensor(doc_embed), torch.tensor(q_idf), qid, id

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('qrels_file', type=str, help="Qrel file in json format")
    parser.add_argument('query_id_file', type=str, help="Embedded query in json format")
    parser.add_argument('docs_id_file', type=str, help="Embedded documents in json format")
    parser.add_argument('idf_file', type=str, help="IDF among documents in json format")
    parser.add_argument('w2v_file', type=str, help="Word2vec model file with npy file under same directory")
    argvs = parser.parse_args()
    word2vec = gensim.models.Word2Vec.load(argvs.w2v_file).wv
    test = DRMMDataset(argvs.qrels_file, argvs.query_id_file, argvs.docs_id_file, argvs.idf_file)
    print('Load data done')
    loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=collate_batch)
    device = torch.device('cpu')

    embedding_weights = torch.FloatTensor(word2vec.vectors)
    word_embedding = nn.Embedding.from_pretrained(embedding_weights).to(device)
    word_embedding.requires_grad = False

    for q, p, n, idf in loader:
        query, pos_doc, neg_doc, q_idf = q.to(device), p.to(device), n.to(device), idf.to(device)
    
        query_mask = (query > 0).float()
        pos_doc_mask = (pos_doc > 0).float()
        neg_doc_mask = (neg_doc > 0).float()
        
        query = word_embedding(query) * query_mask.unsqueeze(-1)
        pos_doc = word_embedding(pos_doc) * pos_doc_mask.unsqueeze(-1)
        neg_doc = word_embedding(neg_doc) * neg_doc_mask.unsqueeze(-1)
        print(query.shape)
        q_len, p_len, n_len = len(q[0]), len(pos_doc[0]), len(neg_doc[0])
        pos_sim = np.zeros(p_len)
        neg_sim = np.zeros(n_len)
        print('=======query=======')
        for i in range(q_len):
            pos_sim += np.array(word2vec.cosine_similarities(query[0][i], pos_doc[0]))
            neg_sim += np.array(word2vec.cosine_similarities(query[0][i], neg_doc[0]))
            print(word2vec.index_to_key[int(q[0][i])], end=' ')
        input()
        pos_sim = (pos_sim / q_len).mean()
        neg_sim = (neg_sim / n_len).mean()
        print('')
        print('=======pos document=======')
        for id in p[0]:
            print(word2vec.index_to_key[int(id)], end=' ')
        
        input()
        print('=======neg document=======')
        for id in n[0]:
            print(word2vec.index_to_key[int(id)], end=' ')
        print('\n', pos_sim, neg_sim)

        input()
