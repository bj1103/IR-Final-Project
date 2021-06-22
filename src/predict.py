import argparse
import json

from torch.utils.data.dataset import Dataset
import gensim.downloader as api
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import rerankDataset
from models.DRMM import DRMM

def collate_batch(batch):
    q, d, q_idf, qids, indexs = zip(*batch)
    q = torch.nn.utils.rnn.pad_sequence(q).T
    d = torch.nn.utils.rnn.pad_sequence(d).T
    q_idf = torch.nn.utils.rnn.pad_sequence(q_idf).T
    return q, d, q_idf, qids, indexs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('ranking_file', type=str, help="Ranjing file in json format")
    parser.add_argument('topics_file', type=str, help="Topic file in json format")
    parser.add_argument('docs_file', type=str, help="Clean document file in json format")
    parser.add_argument('prediction_file', type=str, help="Clean document file in json format")
    parser.add_argument('--model_path', type=str, default='drmm.ckpt', help="Path to model checkpoint")
    parser.add_argument('--valid_steps', type=int, default=1000, help="Steps to validation")
    parser.add_argument('--save_steps', type=int, default=1000, help="Steps to save best model")
    parser.add_argument('--valid_num', type=int, default=200, help="Number of steps doing validation")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--nbins', type=int, default=30, help="Number of bins for histogram")
    argvs = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    print('Loading word2vec model...')
    word2vec = api.load('word2vec-google-news-300')
    embedding_weights = torch.FloatTensor(word2vec.vectors)
    word_embedding = nn.Embedding.from_pretrained(embedding_weights).to(device)
    word_embedding.requires_grad = False
    print('done')
    test_set = rerankDataset(
        argvs.ranking_file,
        argvs.topics_file,
        argvs.docs_file,
        word_model=word2vec,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=argvs.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=4,
    )

    model = DRMM(
        word_embedding=word_embedding,
        embed_dim=embedding_weights.shape[1],
        nbins=argvs.nbins,
        device=device,
    ).to(device)

    prediction = dict()
    for qid in test_set.qids:
        prediction[qid] = [[doc, 0] for doc in test_set.rank_list[qid]]

    ckpt = torch.load(argvs.model_path, map_location=torch.device(device))
    model.load_state_dict(ckpt)
    model.eval()
    for query, document, q_idf, qids, indexs in tqdm(test_loader):
        query, document, q_idf = query.to(device), document.to(device), q_idf.to(device)
        scores = model(query, document, q_idf)
        scores = scores.to('cpu')
        for batch_id, (qid, index) in enumerate(zip(qids, indexs)):
            prediction[qid][index][1] = scores[batch_id].item()
    with open('temp.json', 'w') as out:
        print(json.dumps(prediction, indent=4), file=out)

    for qid in prediction:
        tmp = sorted(prediction[qid], key=lambda pair: pair[1])[::-1]
        prediction[qid] = [pair[0] for pair in tmp]

    with open(argvs.prediction_file, 'w') as out:
        print(json.dumps(prediction, indent=4), file=out)
