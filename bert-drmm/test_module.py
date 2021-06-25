import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import json
from statistics import mean
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn import CrossEntropyLoss
import math
import time
from tqdm.auto import tqdm
from datasets import qrel_dataset
import ir_datasets
from accelerate import Accelerator
import numpy as np
import random
from bertRanker import BertRanker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("./config.json") as f:
    config = json.load(f)

def same_seeds(seed):
      torch.manual_seed(seed)
      if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)
      random.seed(seed)
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True
same_seeds(6553713)

accelerator = Accelerator(fp16=True)
device = accelerator.device

tokenizer = BertTokenizer.from_pretrained(config['bert_model'], do_lower_case = True)


if __name__ == '__main__':
    with open('./qrels.json') as f:
        qrels = json.load(f)

    with open('./id2query_tokenized_id.json') as f:
        id2query = json.load(f)
    
    with open('./id2document_tokenized_id.json') as f:
        id2document = json.load(f)

    train_set = set()
    test_set = set()

    for i in range(1, 2):
        dataset = ir_datasets.load(f"trec-robust04/fold{i}")
        for query in dataset.queries_iter():
            train_set.add(query[0])

    # dataset = ir_datasets.load("trec-robust04/fold5")
    # for query in dataset.queries_iter():
    #     test_set.add(query[0])

    train_data = []
    test_data = []
    train_qrels = []
    test_qrels = []

    print('making dataset...')

    for query_id, document_ids in tqdm(qrels.items()):
        for document_id, label in document_ids['document'].items():
            document = id2document[document_id]
            document_length = len(document)
            for i in range(0, document_length, config['stride']):
                if label > 0:
                    label = 1
                data = [query_id, document_id, label, i]
                if query_id in train_set:
                    train_data.append(data)
                else:
                    test_data.append(data)
        #     break
        # break
                    
    print('train data : ', len(train_data))
    print('test data : ', len(test_data))

    train_dataset = qrel_dataset(id2query, id2document, train_data, config['window_size'], config['max_len'])
    test_dataset = qrel_dataset(id2query, id2document, test_data, config['window_size'], config['max_len'])

    train_dataloader = DataLoader(
                    dataset = train_dataset,
                    batch_size = config["batch_size"],
                    shuffle = True,
                )
    # test_dataloader = DataLoader(
    #                     dataset = test_dataset,
    #                     batch_size = config["batch_size"],
    #                     shuffle = True,
    #                 )

    print('building model...')
    bert, optimizer = build_model()
    bertRanker = BertRanker(bert)

    model_path = f'/tmp2/b07902114/IR-Final-Project/bert/checkpoint/checkpoint.step.1.pt'

    ckpt = torch.load(model_path)
    bert.load_state_dict(ckpt['state_dict'])
    train_loader = iter(trainloader)
    data = next(train_loader)
    bertRanker(data)