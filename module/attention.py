import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

    def attn(self, query, key):
        """
        query: shape=(batch_size, hidden_dim)
        key: shape=(max_seq_len, batch_size, hidden_dim)
        """
        raise NotImplementedError()

    def forward(self, query, key_val):
        """
        query: shape=(batch_size, hidden_dim)
        key_val: shape=(max_seq_len, batch_size, hidden_dim)
        """
        # scores.shape=(batch_size, max_seq_len)
        scores = self.attn(query, key_val)

        # shape=(batch_size, 1, max_seq_len)
        return F.softmax(scores, dim=1).unsqueeze(1)


class DotAttention(BaseAttention):
    def attn(self, query, key):
        return torch.einsum('bh,lbh->bl', query, key)


class GeneralAttention(BaseAttention):
    def __init__(self, hidden_dim):
        super().__init__(hidden_dim)
        self.weight = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def attn(self, query, key):
        energy = self.weight(key)
        return torch.einsum('bh,lbh->bl', query, energy)


class ConcatAttention(BaseAttention):
    def __init__(self, hidden_dim):
        super().__init__(hidden_dim)
        self.weight = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Parameter(torch.randn(hidden_dim))

    def attn(self, query, key):
        """
        query: shape=(batch_size, hidden_dim)
        key: shape=(max_seq_len, batch_size, hidden_dim)
        """
        # expand_query.shape=(max_seq_len, batch_size, hidden_dim)
        expand_query = query.expand(key.size(0), -1, -1)
        # concat_query.shape=(max_seq_len, batch_size, 2 * hidden_dim)
        concat_query = torch.cat([expand_query, key], dim=-1)
        # energy.shape=(max_seq_len, batch_size, hidden_dim)
        energy = self.weight(concat_query).tanh()
        return torch.einsum('h,lbh->bl', self.v, energy)
