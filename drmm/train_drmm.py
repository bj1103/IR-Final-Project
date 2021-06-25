import argparse
from gensim.models import KeyedVectors
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DRMMDataset, collate_batch
from models.DRMM import DRMM

def model_fn(batch, word_embedding, drmm_model, criterion, device):
    query, pos_doc, neg_doc, q_idf = batch
    query, pos_doc, neg_doc, q_idf = query.to(device), pos_doc.to(device), neg_doc.to(device), q_idf.to(device)
    
    query_mask = (query > 0).float()
    pos_doc_mask = (pos_doc > 0).float()
    neg_doc_mask = (neg_doc > 0).float()

    query = word_embedding(query) * query_mask.unsqueeze(-1)
    pos_doc = word_embedding(pos_doc) * pos_doc_mask.unsqueeze(-1)
    neg_doc = word_embedding(neg_doc) * neg_doc_mask.unsqueeze(-1)

    scores_pos = drmm_model(query, query_mask, pos_doc, pos_doc_mask, q_idf)
    scores_neg = drmm_model(query, query_mask, neg_doc, neg_doc_mask, q_idf)
    label = torch.ones(scores_pos.shape).to(device)

    loss = criterion(scores_pos, scores_neg, label)
    acc = len(torch.where(scores_pos > scores_neg)[0]) / len(scores_pos)
    return loss, acc

def valid_fn(dataloader, iterator, word_embedding, model, valid_num, batch_size, device):
    drmm_model.eval()
    running_loss = 0.0
    running_acc = 0.0
    pbar = tqdm(total=valid_num, ncols=0, desc='Valid', unit=' step')
    for i in range(valid_num):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        with torch.no_grad():
            loss, acc = model_fn(batch, word_embedding, model, criterion, device)
            running_loss += loss.item()
            running_acc += acc
        pbar.update()
    running_loss /= valid_num
    running_acc /= valid_num
    pbar.set_postfix(
        loss=f'{running_loss:.4f}',
        acc=f'{running_acc:.4f}',
    )
    pbar.close()
    drmm_model.train()
    return running_loss, running_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('qrels_file', type=str, help="Qrel file in json format")
    parser.add_argument('okapi_qrels_file', type=str, help="Okapi qrel file in json format")
    parser.add_argument('query_id_file', type=str, help="Embedded query in json format")
    parser.add_argument('docs_id_file', type=str, help="Embedded documents in json format")
    parser.add_argument('idf_file', type=str, help="IDF among documents in json format")
    parser.add_argument('w2v_file', type=str, help="Word2vec model file with npy file under same directory")
    parser.add_argument('--model_path', type=str, default='drmm.ckpt', help="Path to model checkpoint")
    parser.add_argument('--valid_steps', type=int, default=1000, help="Steps to validation")
    parser.add_argument('--valid_num', type=int, default=200, help="Number of steps doing validation")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--nbins', type=int, default=30, help="Number of bins for histogram")
    parser.add_argument('--mode', type=str, default='idf', choices=['idf', 'tv'], help="Mode of DRMM")
    argvs = parser.parse_args()
    print(argvs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    print('Loading word2vec model...')
    word2vec = KeyedVectors.load_word2vec_format(argvs.w2v_file, binary=False)

    train_set = DRMMDataset(
        argvs.qrels_file, 
        argvs.okapi_qrels_file, 
        argvs.query_id_file, 
        argvs.docs_id_file,
        argvs.idf_file,
        mode='train',
    )
    train_loader = DataLoader(
        train_set,
        batch_size=argvs.batch_size, 
        shuffle=True, 
        collate_fn=collate_batch,
        num_workers=4,
    )
    test_set = DRMMDataset(
        argvs.qrels_file, 
        argvs.okapi_qrels_file, 
        argvs.query_id_file, 
        argvs.docs_id_file,
        argvs.idf_file,
        mode='test',
    )
    test_loader = DataLoader(
        test_set,
        batch_size=argvs.batch_size, 
        shuffle=False, 
        collate_fn=collate_batch,
        num_workers=4,
    )
    print(f'Train dataset with size {len(train_set)}')
    print(f'Test dataset with size {len(test_set)}')
    train_iterator = iter(train_loader)
    test_iterator = iter(test_loader)

    embedding_weights = torch.FloatTensor(word2vec.vectors)
    word_embedding = nn.Embedding.from_pretrained(embedding_weights).to(device)
    word_embedding.requires_grad = False

    drmm_model = DRMM(
        embed_dim=embedding_weights.shape[1], 
        nbins=argvs.nbins,
        device=device,
        mode=argvs.mode,
    ).to(device)

    optimizer = Adam(drmm_model.parameters(), argvs.lr, weight_decay=argvs.weight_decay)
    criterion = torch.nn.MarginRankingLoss(margin=1, reduction='mean').to(device)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5, verbose=True)

    valid_steps = argvs.valid_steps
    valid_num = argvs.valid_num

    pbar = tqdm(total=valid_steps, ncols=0, desc='Train', unit=' step')
    step = 0
    best_acc = 0.0
    running_acc = 0.0
    running_loss = 0.0

    while True:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, acc = model_fn(batch, word_embedding, drmm_model, criterion, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(drmm_model.parameters(), 5.0)

        optimizer.step()
        optimizer.zero_grad()

        running_acc += acc
        running_loss += loss.item()

        pbar.update()
        pbar.set_postfix(
            step=step + 1,
        )

        if (step + 1) % valid_steps == 0:
            # do validation
            pbar.set_postfix(
                loss=f'{running_loss / argvs.valid_steps:.4f}',
                acc=f'{running_acc / argvs.valid_steps:.4f}',
            )
            # pbar.write(f'loss={running_loss / argvs.valid_steps / argvs.batch_size:.4f}, acc={running_acc / argvs.valid_steps / argvs.batch_size:.4f}')
            pbar.close()
            running_loss = 0.0
            running_acc = 0.0
            
            valid_loss, valid_acc = valid_fn(test_loader, test_iterator, word_embedding, 
                drmm_model, valid_num, argvs.batch_size, device)
            lr_scheduler.step(valid_loss)

            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(drmm_model.state_dict(), argvs.model_path + '_best.ckpt')
                pbar.write(f'Step {step+1}, best model saved with accuracy {best_acc:.4f}')
            torch.save(drmm_model.state_dict(), argvs.model_path + '_last.ckpt')
            
            pbar = tqdm(total=valid_steps, ncols=0, desc='Train', unit=' step')

        step += 1

    pbar.close() 
