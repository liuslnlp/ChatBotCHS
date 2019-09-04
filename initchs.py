from typing import List, Tuple
import argparse
from pathlib import Path
import logging
from util import save_dataset, save_word_dict, create_word_to_ix, create_train_dataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def load_raw_data(path: str) -> List[Tuple[List[str], List[str]]]:
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
    datas.append((list(query), list(reply)))

    return datas


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", default='dataset/xiaohuangji50w_nofenci.conv', type=str)
    parser.add_argument("-o", "--output_dir", default='data', type=str)
    parser.add_argument("--max_seq_len", default=10, type=int)
    parser.add_argument("--max_vocab_size", default=6500, type=int)

    args = parser.parse_args()
    logger.info("Loading raw data...")
    pairs = load_raw_data(args.filename)

    logger.info("Building word dict...")
    word_to_ix = create_word_to_ix(pairs, args.max_seq_len, args.max_vocab_size)
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
