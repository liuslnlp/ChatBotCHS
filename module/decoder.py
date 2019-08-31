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
        scores = self.attn(output.squeeze(0), encoder_outputs)
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
        # output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, sos_flag):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_flag = sos_flag
    
    def forward(self, input_seq, input_length, max_seq_len):
        device = input_seq.device
        batch_size = input_seq.size(1)
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.full((1, batch_size), self.sos_flag, dtype=torch.long, device=device)

        # Initialize tensors to append decoded words to
        all_tokens = []
        all_scores = []

        # Iteratively decode one word token at a time
        for _ in range(max_seq_len):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens.append(decoder_input)
            all_scores.append(decoder_scores)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        all_tokens = torch.cat(all_tokens, 0)
        all_scores = torch.cat(all_scores, 0)
        return all_tokens, all_scores