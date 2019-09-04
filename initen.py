import argparse
import logging
import re
import unicodedata
from pathlib import Path
from util import create_word_to_ix, create_train_dataset, save_word_dict, save_dataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def tokenize(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s.split()


def load_lines(filename):
    lines = {}
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            line_id, *_, text = line.strip().split(" +++$+++ ")
            lines[line_id] = text
    return lines


def load_conversations(filename):
    conversations = []
    with open(filename, 'r', encoding='iso-8859-1') as f:
        for line in f:
            *_, order = line.strip().split(" +++$+++ ")
            order = eval(order)
            conversations.append(order)
    return conversations


def extract_sentence_pairs(conversations, lines):
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation) - 1):
            input_line_id = conversation[i]
            target_line_id = conversation[i + 1]
            input_line = tokenize(lines[input_line_id])
            target_line = tokenize(lines[target_line_id])
            qa_pairs.append((input_line, target_line))
    return qa_pairs


def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--input_dir", default='dataset', type=str)
    parser.add_argument("-o", "--output_dir", default='data', type=str)
    parser.add_argument("--max_seq_len", default=10, type=int)
    parser.add_argument("--max_vocab_size", default=8000, type=int)
    args = parser.parse_args()

    path = Path(args.input_dir)
    corpus_dir = 'cornell movie-dialogs corpus'
    lines_filename = path / corpus_dir / 'movie_lines.txt'
    conversations_filename = path / corpus_dir / 'movie_conversations.txt'

    logger.info("Loading raw data and extracting sentence pairs...")
    lines = load_lines(lines_filename)
    conversations = load_conversations(conversations_filename)
    pairs = extract_sentence_pairs(conversations, lines)

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
