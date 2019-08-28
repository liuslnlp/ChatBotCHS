import torch.nn as nn
import torch.nn.functional as F
import torch


class GRUDecoder(nn.Module):
    def __init__(self, embedding, attn, hidden_dim, output_dim, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = embedding
        embedding_dim = embedding.weight.data.shape[1]
        self.attn = attn
        self.n_layers = n_layers
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.embedding_dropout = nn.Dropout(dropout)
        self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, step, last_hidden, encoder_outputs):
        """
        step: shape=(1, batch_size).
        last_hidden: shape=(n_layers * num_directions, batch_size, hidden_size).
        encoder_outputs: shape=(max_seq_len, batch_size, hidden_size).
        """
        # embedded.shape=(1, batch_size, hidden_dim).
        embedded = self.embedding(step)
        embedded = self.embedding_dropout(embedded)
        # output.shape=(1, batch_size, hidden_dim).
        # hidden.shape=(n_layers * num_directions, batch_size, hidden_size)
        output, hidden = self.gru(embedded, last_hidden)
        # scores.shape=(batch_size, 1, max_seq_len)
        scores = self.attn(output.squeeze(), encoder_outputs)
        # context.shape=(batch_size, 1, hidden_dim)
        context = scores.bmm(encoder_outputs.transpose(0, 1))
        # context.shape=(batch_size, hidden_dim)
        context = context.squeeze(1)
        # output.shape=(batch_size, hidden_dim)
        output = output.squeeze(0)
        # concat.shape=(batch_size, hidden_dim * 2)
        concat = torch.cat((output, context), 1)
        # output.shape=(batch_size, hidden_dim)
        output = torch.tanh(self.concat(concat))
        # output.shape=(batch_size, output_dim)
        output = self.linear(output)
        output = F.log_softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

