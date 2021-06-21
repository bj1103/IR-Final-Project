import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class DRMM(nn.Module):
    def __init__(self, word_embedding: nn.Embedding, embed_dim: int = 300, nbins: int = 30, device: str = 'cuda') -> None:
        super(DRMM, self).__init__()
        self.word_embedding = word_embedding
        self.device = device
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
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Tanh(),
            # nn.Linear(embed_dim, 1),
            # nn.Tanh(),
            # nn.Linear(1, 1),
            # nn.Tanh(),
        )
        # self.gate = nn.Linear(1, 1)
        # self.softmax = nn.Softmax(dim=1)

    def masked_softmax(self, vec: Tensor, mask: Tensor, dim: int = 1, epsilon: float = 1e-5) -> Tensor:
        exps = torch.exp(vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps / masked_sums)

    def forward(self, query: Tensor, document: Tensor, q_idf: Tensor) -> Tensor:
        """Computes relevance score between a query and a document
        Args:
            query: (batch_size, max_query_len, embed_dim) query id sequence
            query_len: [batch_size] query length for each query
            query_mask: [batch_size, max_query_len] mask[i][j] = 0 if query[i][j] is <PAD>, 
                otherwise mask[i][j] = 1
            document: (batch_size, max_document_len, embed_dim) document id sequence
            document_len: [batch_size] document length for each document
        Returns:
            scores: (batch_size) relevance scores
        """

        query_mask = (query > 0).float()
        document_mask = (document > 0).float()

        query_embedding = self.word_embedding(query) * query_mask.unsqueeze(-1)
        document_embedding = self.word_embedding(document) * document_mask.unsqueeze(-1)

        # interaction: (batch, max_query_len, max_document_len)
        query_norm = query_embedding / query_embedding.norm(dim=2).unsqueeze(-1)
        document_norm = document_embedding / document_embedding.norm(dim=2).unsqueeze(-1)
        interaction = torch.bmm(query_norm, document_norm.transpose(1, 2))

        # h.shape: (batch, max_query_len, self.nbins)
        h = torch.empty((interaction.shape[0], interaction.shape[1], self.nbins))
        for b in range(interaction.shape[0]):
            for q in range(interaction.shape[1]):
                h[b, q] = torch.histc(interaction[b, q], bins=self.nbins, min=-1, max=1)
        h = h.to(self.device)
        h = torch.log1p(h)
        # z.shape: (batch, max_query_len)
        z = self.ffn(h).squeeze(-1)

        # g.shape: (batch, max_query_len)
        g = self.gate(query_embedding).squeeze(-1)
        # g = self.softmax(g)
        g = self.masked_softmax(g, query_mask)

        scores = torch.sum(z * g, dim=1)
        return scores
