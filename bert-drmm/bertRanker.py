import torch

class BertRanker(torch.nn.Module):
    def __init__(self, bertClassifier):
        self.bertClassifier = bertClassifier
    
    def forward(self, data):

        result = self.bertClassifier.bert(
                input_ids = data[0], 
                attention_mask = data[1], 
                token_type_ids = data[2]
            )
        print(result)

