from typing import List, Dict, Tuple, Mapping
import torch
import pickle


def convert_tokens_to_ids(tokens: str, word_to_ix: Dict[str, int]) -> List[int]:
    return [word_to_ix.get(token, word_to_ix['[UNK]']) for token in tokens] + [word_to_ix['[EOS]']]


def save_word_dict(word_dict, saved_dir):
    with open(saved_dir / 'vocab.dict', 'wb') as f:
        pickle.dump(word_dict, f)


def save_dataset(queries, replies, lens, saved_dir):
    torch.save(queries, saved_dir / "queries.pt")
    torch.save(replies, saved_dir / "replies.pt")
    torch.save(lens, saved_dir / "lens.pt")


def load_word_dict(saved_dir):
    with open(saved_dir / 'vocab.dict', 'rb') as f:
        word_to_ix = pickle.load(f)
    return word_to_ix


def load_dataset(saved_dir):
    queries = torch.load(saved_dir / "queries.pt")
    replies = torch.load(saved_dir / "replies.pt")
    lens = torch.load(saved_dir / "lens.pt")
    return queries, replies, lens
