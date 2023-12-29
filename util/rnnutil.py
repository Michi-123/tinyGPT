# -*- coding: utf-8 -*-
"""RnnUtil.ipynb

"""

import torch
from torch.utils.data import Dataset


#@title RnnDataset
class RnnDataset(Dataset):
    def __init__(self, corpus_path, max_sequence_length):
        self.corpus = get_corpus(corpus_path)
        self.max_sequence_length = max_sequence_length
        self.source_vocab = self.build_vocab()
        self.vocab = set([c for c in self.corpus])
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_word 

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        source_text = self.corpus[idx]
        source_tokens = self.tokenize(source_text)
        padded_source_tokens = self.pad_sequence(source_tokens, self.max_sequence_length)
        source_indices = self.tokens_to_indices(padded_source_tokens, self.source_vocab)
        return {
            'source_indices': torch.tensor(source_indices),
        }

    def build_vocab(self):
        source_tokens = [token for token in self.corpus]
        source_unique_tokens = set(source_tokens + ['[PAD]'])  # [PAD] を追加
        source_vocab = {token: idx for idx, token in enumerate(source_unique_tokens)}
        return source_vocab


    def pad_sequence(self, tokens, max_length):
        if len(tokens) < max_length:
            padding = ['[PAD]'] * (max_length - len(tokens))
            tokens += padding
        else:
            tokens = tokens[:max_length]
        return tokens

    def tokens_to_indices(self, tokens, vocab):
        return [vocab[token] for token in tokens]



    def get_corpus(self, corpus_path):
        with open(corpus_path, 'rb') as f:
            lines = []
            for line in f:
                line = line.strip().lower().decode("ascii", "ignore")
                if len(line) == 0:
                    continue
                lines.append(line)
        corpus = " ".join(lines)
        return corpus

