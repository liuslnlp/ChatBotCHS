import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import argparse
from pathlib import Path
import logging
from util import load_dataset, load_word_dict
from module import GRUEncoder, AttnGRUDecoder

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
            decoder_input = decoder_input.to(next(decoder.parameters()).device)
        loss += loss_func(decoder_output, targets[t])
    return loss


def save_model(encoder, decoder, dir: str):
    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), output_dir / 'encoder.pkl')
    torch.save(decoder.state_dict(), output_dir / 'decoder.pkl')


def gen_decoder_head(num, sos, device):
    return torch.full((1, num), sos, dtype=torch.long, device=device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default='data', type=str)
    parser.add_argument("-o", "--output_dir", default='output', type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("-m", "--max_seq_len", type=int, default=10)
    parser.add_argument("--embed_dim", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=500)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--encoder_lr", type=float, default=0.0001)
    parser.add_argument("--decoder_lr", type=float, default=0.0005)
    parser.add_argument("--clip", type=float, default=50)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--print_step", type=int, default=20)
    parser.add_argument("--tf_radio", type=float,
                        default=0.97, help='Teacher forcing ratio')

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    return parser.parse_args()


def main():
    args = get_args()
    logger = logging.getLogger(__name__)

    input_dir = Path(args.input_dir)
    logger.info("Loading train data...")
    queries, replies, lens = load_dataset(input_dir)
    word_dict = load_word_dict(input_dir)

    trainset = TensorDataset(queries, replies, lens)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True)

    vocab = len(word_dict)
    embedding = nn.Embedding(vocab, args.embed_dim,
                             padding_idx=word_dict['[PAD]'])
    encoder = GRUEncoder(embedding, args.hidden_dim,
                         args.n_layer, args.dropout)
    decoder = AttnGRUDecoder(embedding, args.hidden_dim,
                             vocab, args.n_layer, args.dropout)
    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')

    for model in (encoder, decoder):
        model.train()
        model.to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.encoder_lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    loss_fct = nn.CrossEntropyLoss(ignore_index=word_dict['[PAD]'])

    logger.info("Training...")
    logger.info(
        f"[Epochs]: {args.epochs}, [Batch Size]: {args.batch_size}, [Max Len]: {args.max_seq_len}")
    logger.info(f"[No CUDA]: {args.no_cuda}")
    logger.info(
        f"[Encoder LR]: {args.encoder_lr}, [Dncoder LR]: {args.decoder_lr}")
    logger.info(
        f"[Embedding Dim]: {args.embed_dim}, [Hidden Dim]: {args.hidden_dim}, [Num Layers]: {args.n_layer}")
    logger.info(
        f"[Dropout Prob]: {args.dropout}, [Teacher Forcing Radio]: {args.tf_radio}, [Clip]: {args.clip}")

    for epoch in range(args.epochs):
        loss_cache = []
        logger.info(f"***** Epoch {epoch} *****")
        for step, batch in enumerate(trainloader):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_ids, targets, lens = tuple(t.t().to(device) for t in batch)
            cur_batch_size = input_ids.shape[1]
            
            encoder_outputs, encoder_hidden = encoder(input_ids, lens[0])
            decoder_input = gen_decoder_head(
                cur_batch_size, word_dict['[SOS]'], device)
            decoder_hidden = encoder_hidden[:decoder.n_layers]
            loss = train_decode_step(
                decoder, decoder_input, decoder_hidden, encoder_outputs, targets, loss_fct, args)
            loss_cache.append(loss.item())
            loss.backward()

            nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
            nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)

            encoder_optimizer.step()
            decoder_optimizer.step()

            if step % args.print_step == 0:
                ave_loss = torch.FloatTensor(loss_cache).mean()
                loss_cache.clear()
                logger.info(
                    f"[epoch]: {epoch}, [batch]: {step}, [average loss]: {ave_loss.item():.6}")
    save_model(encoder, decoder, args.output_dir)


if __name__ == "__main__":
    main()
