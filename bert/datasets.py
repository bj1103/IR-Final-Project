import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pickle
from tqdm import tqdm

with open("./config.json") as f:
    config = json.load(f)
tokenizer = BertTokenizer.from_pretrained(config['bert_model'], do_lower_case = True)   

class qrel_dataset(Dataset):
    def __init__(self, id2query, id2document, data, window_size, max_len, mode='Train'):
        self.data = data
        self.id2query = id2query
        self.id2document = id2document
        self.window_size = window_size
        self.max_len = max_len
        self.mode = mode
    
    def __getitem__(self, idx):
        if self.mode == 'Train':
            query_id = self.data[idx][0]
            document_id = self.data[idx][1]
            label = self.data[idx][2]
            point = self.data[idx][3]
            query = [101] + self.id2query[query_id] + [102]
            document = self.id2document[document_id][point:point+self.window_size] + [102]
            input_ids, token_type_ids, attention_mask = padding(query, document, self.max_len)
            return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids), label
        else:
            query_id = self.data[idx][0]
            document_id = self.data[idx][1]
            point = self.data[idx][2]
            query = [101] + self.id2query[query_id] + [102]
            document = self.id2document[document_id][point:point+self.window_size] + [102]
            input_ids, token_type_ids, attention_mask = padding(query, document, self.max_len)
            return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids), document_id

    def __len__(self):
        return len(self.data)

def padding(query, document, max_seq_len):
    
    padding_len = max_seq_len - len(query) - len(document)
    assert (padding_len >= 0)
    context = query + document
    # context = tokenizer.convert_tokens_to_ids(context)

    input_ids = context + [0] * padding_len
    token_type_ids = [0] * len(query) + [1] * len(document) + [0] * padding_len
    attention_mask = [1] * (len(query) + len(document)) + [0] * padding_len
    return input_ids, token_type_ids, attention_mask

if __name__ == '__main__':

    with open('/tmp2/IR/json_files/qrels.json') as f:
        qrels = json.load(f)

    with open('./id2query_tokenized_id.json') as f:
        id2query = json.load(f)

    with open('./id2document_tokenized_id.json') as f:
        id2document = json.load(f)

    train = set()
    test = set()

    import ir_datasets
    for i in range(1, 5):
        dataset = ir_datasets.load(f"trec-robust04/fold{i}")
        for query in dataset.queries_iter():
            train.add(query[0])

    dataset = ir_datasets.load("trec-robust04/fold5")
    for query in dataset.queries_iter():
        test.add(query[0])

    train_data = []
    test_data = []
    train_qrels = []
    test_qrels = []

    for query_id, document_ids in tqdm(qrels.items()):
        for document_id, label in document_ids['document'].items():
            document = id2document[document_id]
            document_length = len(document)
            for i in range(0, document_length, config['stride']):
                data = [query_id, document_id, label, i]
                if query_id in train:
                    train_data.append(data)
                else:
                    test_data.append(data)
    
    print('train data : ', len(train_data))
    print('test data : ', len(test_data))

    train_dataset = qrel_dataset(id2query, id2document, train_data, config['window_size'], config['max_len'])
    test_dataset = qrel_dataset(id2query, id2document, test_data, config['window_size'], config['max_len'])

    print(train_dataset.__getitem__(0))
    print(train_dataset.__getitem__(1))

    a = tokenizer.convert_ids_to_tokens(train_dataset.__getitem__(0)[0])
    print(a)
    
    b = tokenizer.convert_ids_to_tokens(train_dataset.__getitem__(1)[0])
    print(b)
    


