import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gensim.downloader as api

from dataset import DRMMDataset, collate_batch
from models.DRMM import DRMM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('qrels_file', type=str, help="Qrel file in json format")
    parser.add_argument('topics_file', type=str, help="Topic file in json format")
    parser.add_argument('docs_dir', type=str, help="Doc dir in json format")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--nbins', type=int, default=30, help="Number of bins for histogram")
    argvs = parser.parse_args()

    word2vec = api.load('word2vec-google-news-300')
    embedding_weights = torch.FloatTensor(word2vec.vectors)
    word_embedding = nn.Embedding.from_pretrained(embedding_weights)

    train_set = DRMMDataset(
        argvs.qrels_file, 
        argvs.topics_file, 
        argvs.docs_dir,
        word_model=word2vec,
        mode='train',
    )
    train_loader = DataLoader(
        train_set,
        batch_size=argvs.batch_size, 
        shuffle=True, 
        collate_fn=collate_batch,
    )
    test_set = DRMMDataset(
        argvs.qrels_file, 
        argvs.topics_file, 
        argvs.docs_dir,
        word_model=word2vec,
        mode='test',
    )
    test_loader = DataLoader(
        train_set,
        batch_size=argvs.batch_size, 
        shuffle=False, 
        collate_fn=collate_batch,
    )

    drmm_model = DRMM(
        word_embedding, 
        embed_dim=embedding_weights.shape[0], 
        nbins=argvs.nbins
    )

    for batch in train_loader:
        print(batch)
        input()
