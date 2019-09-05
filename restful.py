from flask import Flask, request
from flask_restful import Resource, Api
from module import GRUEncoder, AttnGRUDecoder, GreedySearchDecoder
import argparse
from util import load_word_dict, convert_tokens_to_ids
from initen import tokenize as en_tokenize
from initchs import tokenize as chs_tokenize
import torch
import torch.nn as nn
from pathlib import Path

app = Flask(__name__)
app.config['JSON_AS_ASCII'] =False
app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))
api = Api(app)

VOCAB_DIR = 'data/chs'
CKPT_DIR = 'output/chs'
MAX_SEQ_LEN = 10
N_LAYER = 2
EMBED_DIM = 500
HIDDEN_DIM = 500

def load_model(encoder, decoder, dir: str):
    output_dir = Path(dir)
    encoder.load_state_dict(torch.load(output_dir / 'encoder.pkl'))
    decoder.load_state_dict(torch.load(output_dir / 'decoder.pkl'))


word_to_ix = load_word_dict(Path(VOCAB_DIR))
vocab = len(word_to_ix)

embedding = nn.Embedding(vocab, EMBED_DIM,
                             padding_idx=word_to_ix['[PAD]'])
encoder = GRUEncoder(embedding, HIDDEN_DIM, N_LAYER)
decoder = AttnGRUDecoder(embedding, HIDDEN_DIM, vocab, N_LAYER)
load_model(encoder, decoder, CKPT_DIR)
searcher = GreedySearchDecoder(encoder, decoder, word_to_ix['[SOS]'])
searcher.eval()
device = torch.device('cpu')
searcher.to(device)
ix_to_word = {idx: word for word, idx in word_to_ix.items()}

def evaluate(searcher, word_to_ix, ix_to_word, sentence, max_seq_len, en: bool):
    device = next(searcher.parameters()).device
    sentence = en_tokenize(sentence) if en else chs_tokenize(sentence)
    input_ids = convert_tokens_to_ids(sentence, word_to_ix)

    lengths = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).t().to(device)
    lengths = torch.tensor(lengths).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens, scores = searcher(input_ids, lengths, max_seq_len)
    decoded_words = [ix_to_word[token.item()] for token in tokens]
    return decoded_words, scores.cpu().tolist()

class ChatBot(Resource):
    def predict(self, query):
        output_tokens, scores = evaluate(searcher, word_to_ix,
                                 ix_to_word, query, MAX_SEQ_LEN, False)
        output_tokens = [x for x in output_tokens if not (
                x == '[EOS]' or x == '[PAD]')]
        reply = ''.join(output_tokens)
        return {'reply': reply, 'scores': scores[:len(reply)]}

    def get(self, query):
        return self.predict(query)

    def put(self, query):
        query = request.form['query']
        return self.predict(query)

api.add_resource(ChatBot, '/<string:query>')

if __name__ == '__main__':
    app.run()