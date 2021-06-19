import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class DRMM(nn.Module):
    def __init__(self, word_embedding: nn.Embedding, embed_dim: int = 300, 
                 nbins: int = 30) -> None:
        super(DRMM, self).__init__()
        self.word_embedding = word_embedding
        self.word_embedding.requires_grad = False
        self.embed_dim = embed_dim

        self.nbins = nbins
        self.cos = nn.CosineSimilarity(dim=3)
        self.ffn = nn.Sequential(
            nn.Linear(nbins, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            nn.Tanh(),
            nn.Linear(1, 1),
            nn.Tanh(),
        )
        self.gate = nn.Linear(embed_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query: Tensor, query_len: list[int], document: Tensor) -> Tensor:
        """Computes relevance score between a query and a document
        Args:
            query: (batch_size, max_query_len) query id sequence
            query_list: [batch_size] query length for each query
            document: (batch_size, max_document_len) document id sequence
        Returns:
            scores: (batch_size) relevance scores
        """

        max_query_len = query.shape[1]
        max_document_len = document.shape[1]

        query = self.word_embedding(query)
        document = self.word_embedding(document)

        query_stack = torch.stack([query] * max_document_len, dim=2)
        # query_stack: (batch, max_query_len, max_document_len, embed)
        document = document.unsqueeze(1)
        # document: (batch, 1, max_document_len, embed)

        interaction = self.cos(query_stack, document)
        # interaction: (batch, max_query_len, max_document_len)

        # h.shape: (batch, max_query_len, self.nbins)
        h = torch.zeros((interaction.shape[0], interaction.shape[1], self.nbins)).to(device)
        for b in range(interaction.shape[0]):
            for q in range(query_len[b]):
                h[b][q] = torch.histc(interaction[b][q], bins=self.nbins, min=-1, max=1)
        h = torch.log1p(h)

        # z.shape: (batch, max_query_len)
        z = self.ffn(h).squeeze()
        # g.shape: (batch, max_query_len)
        g = self.gate(query).squeeze()
        g = self.softmax(g)

        scores = torch.sum(z * g, dim=1)
        return scores
