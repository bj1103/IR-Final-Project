import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class CedrDRMM(nn.Module):
    def __init__(self, bert_encoder: nn.Module, embed_dim: int = 300, nbins: int = 30, device: str = 'cuda') -> None:
        super(CedrDRMM, self).__init__()
        self.device = device
        self.nbins = nbins
        self.drmm = DRMM(embed_dim=embed_dim, nbins=nbins, device=device)
        self.bert_encoder = bert_encoder

    def forward(input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor, labels: Tensor):
        cls_reps, query_reps, doc_reps = self.bert_encoder(input_ids, attention_mask, token_type_ids, labels)
        scores = self.drmm(query_reps, query_mask, doc_reps, doc_mask)
        return scores

class DRMM(nn.Module):
    def __init__(self, embed_dim: int = 300, nbins: int = 30, device: str = 'cuda') -> None:
        super(DRMM, self).__init__()
        self.device = device
        self.nbins = nbins
        self.cos = nn.CosineSimilarity(dim=3)
        self.ffn = nn.Sequential(
            nn.Linear(nbins, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            nn.Tanh(),
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, 1),
        )

    """Thanks to https://discuss.pytorch.org/t/apply-mask-softmax/14212"""
    def masked_softmax(self, vec: Tensor, mask: Tensor, dim: int = 1, epsilon: float = 1e-5) -> Tensor:
        exps = torch.exp(vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps / masked_sums)

    def forward(self, query: Tensor, query_mask: Tensor, document: Tensor, 
                document_mask: Tensor, q_idf: Tensor) -> Tensor:
        """Computes relevance score between a query and a document
        Args:
            query: (batch_size, max_query_len, embed_dim) query id sequence
            query_mask: [batch_size, max_query_len] mask[i][j] = 0 if query[i][j] is <PAD>, 
                otherwise mask[i][j] = 1
            document: (batch_size, max_document_len, embed_dim) document id sequence
            document_mask: [batch_size, max_document_len] mask[i][j] = 0 if query[i][j] is <PAD>, 
                otherwise mask[i][j] = 1
            query_idf: (batch_size, max_query_len) IDF of each query term
        Returns:
            scores: (batch_size) relevance scores
        """

        # interaction: (batch, max_query_len, max_document_len)
        query_norm = query / query.norm(dim=2).unsqueeze(-1)
        document_norm = document / document.norm(dim=2).unsqueeze(-1)
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

        gete_input = q_idf.float().unsqueeze(-1) if self.mode == 'idf' else query
        # g.shape: (batch, max_query_len)
        g = self.gate(gete_input).squeeze(-1)
        # g = self.softmax(g)
        g = self.masked_softmax(g, query_mask)

        scores = torch.sum(z * g, dim=1)
        return scores
