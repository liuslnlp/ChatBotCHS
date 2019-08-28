import torch
import torch.nn as nn
from pathlib import Path
from module import GRUEncoder, GRUDecoder, GreedySearchDecoder, DotAttention
import argparse
from util import load_word_dict


def evaluate(encoder, searcher, word_to_ix, ix_to_word, sentence, device, max_seq_len):
    encoder.eval()
    searcher.eval()
    input_ids = [word_to_ix.get(word, word_to_ix['[UNK]']) for word in sentence]
    lengths = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).t().to(device)
    lengths = torch.tensor(lengths).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens, scores = searcher(input_ids, lengths, max_seq_len)
    decoded_words = [ix_to_word[token.item()] for token in tokens]
    return decoded_words

def load_model(encoder, decoder, dir:str):
    output_dir = Path(dir)
    encoder.load_state_dict(torch.load(output_dir / 'encoder.pkl'))
    decoder.load_state_dict(torch.load(output_dir / 'decoder.pkl'))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='data', type=str)
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--max_seq_len", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--tf_radio", type=float, default=0.5, help='teacher_forcing_ratio')

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    return parser.parse_args()

def main():
    args = get_args()
    word_to_ix = load_word_dict(Path(args.input_dir))
    vocab = len(word_to_ix)
    ix_to_word = {v:k for k, v in word_to_ix.items()}
    embedding = nn.Embedding(vocab, args.embed_dim, padding_idx=word_to_ix['[PAD]'])
    encoder = GRUEncoder(embedding, args.hidden_dim)
    attn = DotAttention(args.hidden_dim)
    decoder = GRUDecoder(embedding, attn, args.hidden_dim, vocab)
    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')
    searcher = GreedySearchDecoder(encoder, decoder, word_to_ix['[SOS]'])
    query = '天气不错'
    reply = evaluate(encoder, searcher, word_to_ix, ix_to_word, query, device, args.max_seq_len)
    print(f'Bot: {"".join(reply)}')
if __name__ == "__main__":
    main()