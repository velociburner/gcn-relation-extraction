from csv import DictReader
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

import deps


class SemEvalDataset(Dataset):

    def __init__(self, path, labels, nlp, vocab=None):
        self.columns = ["relation", "e1", "e2", "sentence"]
        self.label_map = {label: i for i, label in enumerate(labels)}
        self.pad = "<pad>"
        self.unk = "<unk>"

        self.e1_start = "<e1>"
        self.e1_end = "</e1>"
        self.e2_start = "<e2>"
        self.e2_end = "</e2>"

        self.sentences, self.sentence_tokens, self.labels = self._get_data(path)
        self.adj_matrices = deps.generate_matrices(self.sentences, nlp)

        # only initialize vocab for training set
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = build_vocab_from_iterator(
                self.tokens, specials=[self.pad, self.unk]
            )
            self.vocab.set_default_index(self.vocab[self.unk])

    def _get_data(self, path):
        sentences: list[str] = []
        sentence_tokens: list[list[str]] = []
        labels: list[str] = []
        with open(path, 'r', encoding='utf8') as f:
            for line in DictReader(f, fieldnames=self.columns, delimiter="\t"):
                # sentences with entity at index 0 start with a space
                sentence = line["sentence"].strip()
                tokens = sentence.split(" ")
                label = line["relation"]

                # to skip all examples with class "Other", since it's common
                # if label == "Other":
                #     continue

                sentences.append(sentence)
                sentence_tokens.append(tokens)
                labels.append(label)

        return sentences, sentence_tokens, labels

    @property
    def tokens(self):
        for tokens in self.sentence_tokens:
            yield tokens

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        x = torch.tensor(self.vocab(self.sentence_tokens[idx]))
        y = torch.tensor(self.label_map[self.labels[idx]])
        adj_matrix = self.adj_matrices[idx]
        return x, y, adj_matrix


def collate_fn(examples: Iterable[tuple], padding_value: float = 0.):
    """Function used as collate_fn for PyTorch Dataloaders. It simply pads each
    sequence to the longest sequence length in the batch and returns it, along
    with a tensor of labels and padded adjacency matrix of dependency arcs."""
    token_indices = nn.utils.rnn.pad_sequence(
        [example[0] for example in examples],
        batch_first=True,
        padding_value=padding_value
    )
    labels = torch.tensor([example[1] for example in examples])

    adj_matrices = [example[2] for example in examples]
    max_size = max([matrix.size()[0] for matrix in adj_matrices])

    # pad each adjacency matrix to the size of the largest one
    # each matrix is padded along both dimensions and remains a square matrix
    padded_matrices = torch.stack([
        F.pad(
            matrix,
            (0, max_size - matrix.size()[1], 0, max_size - matrix.size()[0]),
            value=padding_value
        )
        for matrix in adj_matrices
    ])

    return token_indices, labels, padded_matrices
