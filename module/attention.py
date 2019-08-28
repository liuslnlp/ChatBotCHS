import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
    
    def attn(self, query, key):
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