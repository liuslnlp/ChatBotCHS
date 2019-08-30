from typing import List, Dict, Tuple
import argparse
import torch
from pathlib import Path
import logging
from util import save_dataset, save_word_dict, convert_tokens_to_ids
from collections import defaultdict
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def load_raw_data(path: str) -> List[Tuple[str, str]]:
    path = Path(path)
    with path.open(encoding='utf-8') as f:
        raw_datas = f.readlines()
    datas = []
    query_buf = []
    reply_buf = []
    flag = 0
    for line in raw_datas:
        line = line.strip()

        if line == 'E' and query_buf:
            flag = 0
            query = ','.join(query_buf).replace(' ', '')
            reply = ','.join(reply_buf).replace(' ', '')
            datas.append((query, reply))
            query_buf.clear()
            reply_buf.clear()
        elif line[:2] == 'M ':
            if flag == 0:
                flag = 1
                line = line[2:]
            elif flag == 1:
                flag = 2
                line = line[2:]

        if flag == 1:
            query_buf.append(line)
        elif flag == 2:
            reply_buf.append(line)

    query = ','.join(query_buf).replace(' ', '')
    reply = ','.join(reply_buf).replace(' ', '')
    datas.append((query, reply))

    return datas


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def create_word_to_ix(tokens: List[Tuple[str, str]], max_vocab_size: int):
    word_to_ix = {'[PAD]': 0, '[SOS]': 1, '[EOS]': 2, '[UNK]': 3}
    max_vocab_size -= 4
    freq_dict = defaultdict(int)
    for pair in tokens:
        for sent in pair:
            for token in sent:
                freq_dict[token] += 1
    sorted_items = sorted(freq_dict.items(), key=lambda t: t[1], reverse=True)[
        :max_vocab_size]
    for word, _ in sorted_items:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def create_train_dataset(datas: List[Tuple[str, str]], word_to_ix: Dict[str, int], max_seq_len: int):
    total = len(datas)
    queries = torch.full((total, max_seq_len),
                         word_to_ix['[PAD]'], dtype=torch.long)
    replies = torch.full((total, max_seq_len),
                         word_to_ix['[PAD]'], dtype=torch.long)
    lens = torch.zeros((total, 2), dtype=torch.long)
    for i, (query, reply) in enumerate(datas):
        fix_len = min(max_seq_len - 1, len(query))
        query = torch.tensor(convert_tokens_to_ids(
            query[:fix_len], word_to_ix))
        queries[i, :fix_len + 1] = query
        lens[i, 0] = fix_len + 1

        fix_len = min(max_seq_len - 1, len(reply))
        reply = torch.tensor(convert_tokens_to_ids(
            reply[:fix_len], word_to_ix))
        replies[i, :fix_len + 1] = reply
        lens[i, 1] = fix_len + 1

    return queries, replies, lens


def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", default='dataset/xiaohuangji50w_nofenci.conv', type=str)
    parser.add_argument("--output_dir", default='data', type=str)
    parser.add_argument("--max_seq_len", default=32, type=int)
    parser.add_argument("--max_vocab_size", default=6500, type=int)

    args = parser.parse_args()
    logger.info("Loading raw data...")
    pairs = load_raw_data(args.path)

    logger.info("Building word dict...")
    word_to_ix = create_word_to_ix(pairs, args.max_vocab_size)
    logger.info(f"Vocab size: {len(word_to_ix)}")

    logger.info("Building tensor-format dataset...")
    queries, replies, lens = create_train_dataset(
        pairs, word_to_ix, args.max_seq_len)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving...")
    save_word_dict(word_to_ix, output_dir)
    save_dataset(queries, replies, lens, output_dir)

    logger.info("All Done!")


if __name__ == "__main__":
    main()
