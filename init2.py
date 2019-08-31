from typing import List, Dict, Tuple
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
import argparse
import torch
import logging
from util import save_dataset, save_word_dict, convert_tokens_to_ids
from collections import defaultdict
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_lines(fileName):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            line_id, *_, text = line.strip().split(" +++$+++ ")
            lines[line_id] = text
    return lines

def load_conversations(fileName):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
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
            target_line_id = conversation[i+1]
            input_line = tokenize(lines[input_line_id]) 
            target_line = tokenize(lines[target_line_id]) 
            qa_pairs.append((input_line, target_line))
    return qa_pairs

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

def create_word_to_ix(tokens: List[Tuple[List[str], List[str]]], max_vocab_size: int):
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

def create_train_dataset(datas: List[Tuple[List[str], List[str]]], word_to_ix: Dict[str, int], max_seq_len: int):
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
    parser.add_argument("--input_dir", default='dataset', type=str)
    parser.add_argument("--output_dir", default='data', type=str)
    parser.add_argument("--max_seq_len", default=32, type=int)
    parser.add_argument("--max_vocab_size", default=40000, type=int)
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