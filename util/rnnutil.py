#Debug
# -*- coding: utf-8 -*-
"""RnnUtil.ipynb

"""

import torch
from torch.utils.data import Dataset, DataLoader


#@title RnnDataset
class RnnDataset(Dataset):
    def __init__(self, corpus_path, max_sequence_length):
        self.corpus = self.read_corpus(corpus_path)
        self.max_sequence_length = max_sequence_length

        vocab = set([c for c in self.corpus]) # 一覧
        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}
        self.index_to_word = {index: word for index, word in enumerate(vocab)}
        
        input_corpus, target_corpus = self.corpus_set()
        self.input_corpus = input_corpus
        self.target_corpus = target_corpus

    def corpus_set(self):
        seq_size = self.max_sequence_length
        step = 1
        input_corpus, target_corpus = [], []
        # Convert the data into a series of different SEQLEN-length subsequences.
        for i in range(0, len(self.corpus) - seq_size, step):
            end_of_corpus = i + seq_size
            input_corpus.append(self.corpus[i: end_of_corpus])
            target_corpus.append(self.corpus[end_of_corpus]) #次の一文字
        return input_corpus, target_corpus

    def __len__(self):
        return len(self.input_corpus)

    def __getitem__(self, idx):
        source_indices = self.corpus_to_indices(self.input_corpus[idx], self.word_to_index)
        target_indices = self.corpus_to_indices(self.target_corpus[idx], self.word_to_index)
        return {
            'source_indices': torch.tensor(source_indices),
            'target_indices': torch.tensor(target_indices),
        }

    def corpus_to_indices(self, corpus, word_to_index):
        return [word_to_index[letter] for letter in corpus]

    def read_corpus(self, corpus_path):
        with open(corpus_path, 'rb') as f:
            lines = []
            for line in f:
                line = line.strip().lower().decode("ascii", "ignore")
                if len(line) == 0:
                    continue
                lines.append(line)
        corpus = " ".join(lines)
        return corpus

    def indices_to_sequence(self, indices):
        sequence = ''
        for index in indices:
            letter = self.index_to_word[index]
            sequence += letter
        return sequence