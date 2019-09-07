from flask import Flask, request
from flask_restful import Resource, Api
from module import GRUEncoder, AttnGRUDecoder, GreedySearchDecoder
from util import load_word_dict, convert_tokens_to_ids
from initen import tokenize as en_tokenize
from initchs import tokenize as chs_tokenize
from eval import load_model, evaluate
import argparse
import torch
import torch.nn as nn
from pathlib import Path

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config.update(RESTFUL_JSON=dict(ensure_ascii=False))
api = Api(app)

VOCAB_DIR = 'data/chs'
CKPT_DIR = 'output/chs'
MAX_SEQ_LEN = 10
N_LAYER = 2
EMBED_DIM = 500
HIDDEN_DIM = 500


word_to_ix = load_word_dict(Path(VOCAB_DIR))
vocab = len(word_to_ix)

embedding = nn.Embedding(vocab, EMBED_DIM,
                         padding_idx=word_to_ix['[PAD]'])
encoder = GRUEncoder(embedding, HIDDEN_DIM, N_LAYER)
decoder = AttnGRUDecoder(embedding, HIDDEN_DIM, vocab, N_LAYER)
device = torch.device('cpu')
load_model(encoder, decoder, CKPT_DIR, device)
searcher = GreedySearchDecoder(encoder, decoder, word_to_ix['[SOS]'])
searcher.eval()
searcher.to(device)
ix_to_word = {idx: word for word, idx in word_to_ix.items()}


class ChatBot(Resource):
    def predict(self, query):
        output_tokens, scores = evaluate(searcher, word_to_ix,
                                         ix_to_word, query, MAX_SEQ_LEN, False)
        scores = scores.cpu().tolist()
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


@app.route('/')
def index():
    html = """<h1>ChatBot RESTful API</h1> GET: URL/... <br /> PUT: {'query': ...}"""
    return html


if __name__ == '__main__':
    app.run(host='0.0.0.0')
