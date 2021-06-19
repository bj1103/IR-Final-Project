import argparse
import gensim.downloader as api
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DRMMDataset, collate_batch
from models.DRMM import DRMM

def loss_fn(scores_pos: Tensor, scores_neg: Tensor) -> Tensor:
    z = torch.zeros(scores_pos.shape)
    return torch.max(z, 1.0 - scores_pos + scores_neg)

def model_fn(batch, model, device):
    query, pos_doc, neg_doc, query_len = batch
    query, pos_doc, neg_doc = query.to(device), pos_doc.to(device), neg_doc.to(device)
    scores_pos = drmm_model(query, pos_doc, query_len)
    scores_neg = drmm_model(query, neg_doc, query_len)
    loss = loss_fn(scores_pos, scores_neg)
    return loss

def valid_fn(dataloader, model, valid_num, device):
    drmm_model.eval()
    running_loss = 0.0
    pbar = tqdm(total=valid_num, ncols=0, desc='Valid', unit=' step')
    for i in range(valid_num):
        try:
            batch = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_iterator)
            batch = next(test_iterator)
        with torch.no_grad():
            loss = model_fn(batch, model, device)
            running_loss += loss.item()
        pbar.update()
        pbar.set_postfix(
            loss=f'{running_loss / (i+1):.2f}',
        )
    pbar.close()
    drmm_model.train()
    return running_loss / valid_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('qrels_file', type=str, help="Qrel file in json format")
    parser.add_argument('topics_file', type=str, help="Topic file in json format")
    parser.add_argument('docs_dir', type=str, help="Doc dir in json format")
    parser.add_argument('--model_path', type=str, default='drmm.ckpt', help="Path to model checkpoint")
    parser.add_argument('--valid_steps', type=int, default=1000, help="Steps to validation")
    parser.add_argument('--save_steps', type=int, default=1000, help="Steps to save best model")
    parser.add_argument('--valid_num', type=int, default=50, help="Number of steps doing validation")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--nbins', type=int, default=30, help="Number of bins for histogram")
    argvs = parser.parse_args()

    print('Loading word2vec model...')
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
    print(f'Train loader with size {len(train_loader)}')
    print(f'Test loader with size {len(test_loader)}')
    train_iterator = iter(train_loader)
    test_iterator = iter(test_loader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    drmm_model = DRMM(
        word_embedding, 
        embed_dim=embedding_weights.shape[0], 
        nbins=argvs.nbins,
        device=device,
    ).to(device)
    optimizer = AdamW(drmm_model.parameters(), argvs.lr)

    valid_steps = argvs.valid_steps
    save_steps = argvs.save_steps
    valid_num = argvs.valid_num
    pbar = tqdm(total=valid_steps, ncols=0, desc='Train', unit=' step')
    step = 0
    min_loss = float('inf')
    prev_loss = float('int')
    best_state_dict = None

    while True:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss = model_fn(batch, model, device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.update()
        pbar.set_postfix(
            loss=f'f{loss.item():.3f}',
            step=step + 1,
        )

        if (step + 1) % valid_steps == 0:
            # do validation
            pbar.close()
            valid_loss = valid_fn(test_loader, drmm_model, valid_num, device)

            if valid_loss < min_loss:
                min_loss = valid_loss
                best_state_dict = drmm_model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc='Train', unit=' step')

        if (step + 1) % save_steps == 0:
            if prev_loss - min_loss < 1e-5:
                break 
            torch.save(best_state_dict, argvs.model_path)
            prev_loss = min_loss
            pbar.write(f'Step {step+1}, best model saved with loss {min_loss:.3f}')

        step += 1

    pbar.close() 
