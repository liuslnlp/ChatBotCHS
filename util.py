from typing import List, Dict, Tuple
import torch
import pickle
from pathlib import Path
from collections import defaultdict


def convert_tokens_to_ids(tokens: List[str], word_to_ix: Dict[str, int]) -> List[int]:
    return [word_to_ix.get(token, word_to_ix['[UNK]']) for token in tokens] + [word_to_ix['[EOS]']]


def save_word_dict(word_dict: Dict[str, int], saved_dir: Path) -> None:
    with open(saved_dir / 'vocab.dict', 'wb') as f:
        pickle.dump(word_dict, f)


def load_word_dict(saved_dir: Path):
    with open(saved_dir / 'vocab.dict', 'rb') as f:
        word_to_ix = pickle.load(f)
    return word_to_ix


def save_dataset(queries, replies, lens, saved_dir):
    torch.save(queries, saved_dir / "queries.pt")
    torch.save(replies, saved_dir / "replies.pt")
    torch.save(lens, saved_dir / "lens.pt")


def load_dataset(saved_dir: Path):
    queries = torch.load(saved_dir / "queries.pt")
    replies = torch.load(saved_dir / "replies.pt")
    lens = torch.load(saved_dir / "lens.pt")
    return queries, replies, lens


def create_word_to_ix(tokens: List[Tuple[List[str], List[str]]], max_seq_len: int, max_vocab_size: int):
    word_to_ix = {'[PAD]': 0, '[SOS]': 1, '[EOS]': 2, '[UNK]': 3}
    max_vocab_size -= 4
    freq_dict = defaultdict(int)
    for pair in tokens:
        for sent in pair:
            for token in sent[:max_seq_len]:
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
        query = convert_tokens_to_ids(query[:max_seq_len - 1], word_to_ix)
        query = torch.tensor(query)
        fix_len = query.shape[0]
        queries[i, :fix_len] = query
        lens[i, 0] = fix_len

        reply = convert_tokens_to_ids(reply[:max_seq_len - 1], word_to_ix)
        reply = torch.tensor(reply)
        fix_len = reply.shape[0]
        replies[i, :fix_len] = reply
        lens[i, 1] = fix_len
    return queries, replies, lens
