import torch
import torch.nn as nn
from pathlib import Path
from module import GRUEncoder, GRUDecoder, GreedySearchDecoder, DotAttention
import argparse
from util import load_word_dict, convert_tokens_to_ids
from init2 import tokenize


def evaluate(searcher, word_to_ix, ix_to_word, sentence, max_seq_len):
    device = next(searcher.parameters()).device
    ###################
    sentence = tokenize(sentence)
    ###################
    input_ids = convert_tokens_to_ids(sentence, word_to_ix)

    lengths = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).t().to(device)
    lengths = torch.tensor(lengths).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens, scores = searcher(input_ids, lengths, max_seq_len)
    print(scores)
    decoded_words = [ix_to_word[token.item()] for token in tokens]
    return decoded_words


def evaluate_loop(searcher, word_to_ix, max_seq_len):
    ix_to_word = {idx: word for word, idx in word_to_ix.items()}
    while True:
        input_sentence = input('> ')
        if input_sentence in ('q', 'quit'):
            break
        output_tokens = evaluate(searcher, word_to_ix,
                                 ix_to_word, input_sentence, max_seq_len)
        output_tokens = [x for x in output_tokens if not (
            x == '[EOS]' or x == '[PAD]')]      
        print('Bot:', ' '.join(output_tokens))


def load_model(embedding, encoder, decoder, dir: str):
    output_dir = Path(dir)
    encoder.load_state_dict(torch.load(output_dir / 'encoder.pkl'))    
    decoder.load_state_dict(torch.load(output_dir / 'decoder.pkl'))
    embedding.load_state_dict(torch.load(output_dir / 'embedding.pkl'))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_dir", default='data', type=str)
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--max_seq_len", type=int, default=10)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=500)

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    return parser.parse_args()


def main():
    args = get_args()
    word_to_ix = load_word_dict(Path(args.vocab_dir))
    vocab = len(word_to_ix)

    word_to_ix.update({'[PAD]': 0, '[SOS]': 1, '[EOS]': 2, '[UNK]': vocab})




    embedding = nn.Embedding(vocab, args.embed_dim,
                             padding_idx=word_to_ix['[PAD]'])
    encoder = GRUEncoder(embedding, args.hidden_dim, args.n_layer)
    attn = DotAttention(args.hidden_dim)
    decoder = GRUDecoder(embedding, args.hidden_dim, vocab, args.n_layer)
    load_model(embedding, encoder, decoder, args.output_dir)
    searcher = GreedySearchDecoder(encoder, decoder, word_to_ix['[SOS]'])
    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')
    searcher.eval()
    searcher.to(device)
    evaluate_loop(searcher, word_to_ix, args.max_seq_len)


if __name__ == "__main__":
    main()
