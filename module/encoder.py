import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, embedding, hidden_dim, n_layers=1, dropout=0):
        super().__init__()
        self.embedding = embedding
        self.hidden_dim = hidden_dim
        embedding_dim = embedding.weight.data.shape[1]
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=(
            0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, seqs, lens, hidden=None):
        """
        seqs: shape=(max_seq_len, batch_size).
        lens: shape=(batch_size, ).
        """
        # embedded.shape = shape=(max_seq_len, batch_size, embed_dim).
        embedded = self.embedding(seqs)
        # print(f"Encoder embed shape: {embedded.shape}, lens shape: {lens.shape}")
        # l1 = (seqs.shape[0] - (seqs == 0).sum(dim=0))
        # print(l1 - lens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lens, enforce_sorted=False)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_dim] + \
            outputs[:, :, self.hidden_dim:]
        # Return output and final hidden state

        # output.shape=(max_length, batch_size, hidden_size)
        # hidden.shape=(n_layers * num_directions, batch_size, hidden_size)
        return outputs, hidden
