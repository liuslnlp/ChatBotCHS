import torch.nn as nn
import torch


class GreedySearchDecoder(nn.Module):
    """Greedy Search, always used in evaluation.
    """

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
            decoder_scores, decoder_input = torch.max(decoder_output.softmax(dim=-1), dim=1)
            # Record token and score
            all_tokens.append(decoder_input)
            all_scores.append(decoder_scores)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        all_tokens = torch.cat(all_tokens, 0)
        all_scores = torch.cat(all_scores, 0)
        return all_tokens, all_scores
