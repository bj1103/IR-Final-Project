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

def build_model():
    model = BertForSequenceClassification.from_pretrained(config['bert_model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], amsgrad=True)
    model.to(device)
    return model, optimizer

def dev(model, devloader):
    model.eval()
    with torch.no_grad():
        dev_loss = []
        for step, data in enumerate(devloader):
            data = [x.to(device) for x in data]
            output = model(
                input_ids = data[0], 
                attention_mask = data[1], 
                token_type_ids = data[2], 
                labels = data[3]
            )
            # print('step : ', step, ' dev loss : ', loss.item(), end='\r')
            dev_loss.append(output.loss.item())
            if step > 1000:
                break
        return dev_loss

def train(model, optimizer, scheduler, trainloader, devloader, total_step):
    for epoch in range(1, config["epoch"] + 1):
        print('epoch:', epoch)
        model.train()
        train_loss = []
        epoch_start_time = time.time()
        train_loader = iter(trainloader)
        print(len(trainloader))
        for step in tqdm(range(total_step)):
            data = next(train_loader)
            data = [x.to(device) for x in data]

            output = model(
                input_ids = data[0], 
                attention_mask = data[1], 
                token_type_ids = data[2], 
                labels = data[3]
            )
            # print('step : ', step, ' train loss : ', output.loss.item(), end='\r')

            train_loss.append(output.loss.item())

            accelerator.backward(output.loss)

            if (step % config['accum_step'] == 0) or (step + 1 == len(trainloader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            optimizer.step()
            optimizer.zero_grad()

            if step % config['save_step'] == 1 and step > config['save_step']:
                dev_loss = dev(model, devloader)
                print('[%03d/%03d] %2.2f sec(s) Tarin loss: %3.6f Dev loss: %3.6f' % (epoch , config["epoch"], time.time() - epoch_start_time, mean(train_loss), mean(dev_loss)))
                c = step//config['save_step']
                checkpoint_path = f'/tmp2/b07902114/IR-Final-Project/bert/checkpoint_2/checkpoint.step.{c}.pt'
                torch.save({'state_dict' : model.state_dict(), 'epoch' : epoch, }, checkpoint_path)
                print('')
                model.train()
            if step > total_step:
                break
        dev_loss = dev(model, devloader)
        print('[%03d/%03d] %2.2f sec(s) Tarin loss: %3.6f Dev loss: %3.6f' % (epoch , config["epoch"], time.time() - epoch_start_time, mean(train_loss), mean(dev_loss)))
        checkpoint_path = f'/tmp2/b07902114/IR-Final-Project/bert/checkpoint_2/checkpoint.{epoch}.pt'
        torch.save({'state_dict' : model.state_dict(), 'epoch' : epoch, }, checkpoint_path)
        model.train()
    
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
  
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)



if __name__ == '__main__':
    print('loading data...')

    with open('./qrels.json') as f:
        qrels = json.load(f)

    with open('./id2query_tokenized_id.json') as f:
        id2query = json.load(f)
    
    with open('./id2document_tokenized_id.json') as f:
        id2document = json.load(f)

    train_set = set()
    test_set = set()

    for i in range(1, 5):
        dataset = ir_datasets.load(f"trec-robust04/fold{i}")
        for query in dataset.queries_iter():
            train_set.add(query[0])

    dataset = ir_datasets.load("trec-robust04/fold5")
    for query in dataset.queries_iter():
        test_set.add(query[0])

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
    test_dataloader = DataLoader(
                        dataset = test_dataset,
                        batch_size = config["batch_size"],
                        shuffle = True,
                    )

    print('building model...')
    model, optimizer = build_model()
    # total_step = len(train_dataloader) * config['epoch'] / config['accum_step']
    total_step = 200000
    scheduler = get_cosine_schedule_with_warmup(optimizer, config['warmup_step'], total_step)

    print('start training...')
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    train(model, optimizer, scheduler, train_dataloader, test_dataloader, total_step)
    

