import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import argparse
from pathlib import Path
import logging
from util import load_dataset, load_word_dict
from module import GRUEncoder, GRUDecoder, DotAttention
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='data', type=str)
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--print_step", type=int, default=20)
    parser.add_argument("--tf_radio", type=float, default=0.5, help='teacher_forcing_ratio')

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    return parser.parse_args()


def train_decode_step(decoder, decoder_input, decoder_hidden, encoder_outputs, targets, loss_func, args):
    use_teacher_forcing = True if random.random() < args.tf_radio else False
    loss = 0
    
    for t in range(args.max_seq_len):
        decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
        if use_teacher_forcing:
            decoder_input = targets[t].view(1, -1)
        else:
            _, topi = decoder_output.topk(1) 
            decoder_input = topi[:decoder_input.shape[1], 0].unsqueeze(0)
            # decoder_input = torch.LongTensor([[topi[i][0] for i in range(decoder_input.shape[1])]])
            # print(decoder_input.shape) #1,64 64
            decoder_input = decoder_input.to(next(decoder.parameters()).device)
        loss += loss_func(decoder_output, targets[t])
    return loss

def save_model(encoder, decoder, dir:str):
    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), output_dir / 'encoder.pkl')
    torch.save(decoder.state_dict(), output_dir / 'decoder.pkl')


def main():
    args = get_args()
    input_dir = Path(args.input_dir)
    logger = logging.getLogger(__name__)
    target_pad = 0
    loss_fct = nn.NLLLoss(ignore_index=target_pad)
    # loss_fct(logits.view(-1, self.model.tag_size), tag_ids.view(-1))

    queries, replies, lens = load_dataset(input_dir)
    word_dict = load_word_dict(input_dir)
    trainset = TensorDataset(queries, replies, lens)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True)

    vocab = len(word_dict)
    embedding = nn.Embedding(vocab, args.embed_dim, padding_idx=word_dict['[PAD]'])
    encoder = GRUEncoder(embedding, args.hidden_dim)
    attn = DotAttention(args.hidden_dim)
    decoder = GRUDecoder(embedding, attn, args.hidden_dim, vocab)
    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')
    for model in (encoder, decoder):
        model.train()
        model.to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    # print(args.max_seq_len)
    for epoch in range(args.epochs):
        logger.info(f"***** Epoch {epoch} *****")
        for step, batch in enumerate(trainloader):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_ids, targets, lens = tuple(t.t().to(device) for t in batch)
            true_batch_size = input_ids.shape[1]
            decoder_input = torch.full((true_batch_size, ), word_dict['[SOS]'], dtype=torch.long).unsqueeze(0).to(device)
            encoder_outputs, encoder_hidden = encoder(input_ids, lens[0])
            decoder_hidden = encoder_hidden[:decoder.n_layers]
            loss = train_decode_step(decoder, decoder_input, decoder_hidden, encoder_outputs, targets, loss_fct, args)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            if step % args.print_step == 0:
                logger.info(
                    f"[epoch]: {epoch}, [batch]: {step}, [loss]: {loss.item():.6}")
    save_model(encoder, decoder, args.output_dir)

if __name__ == "__main__":
    main()