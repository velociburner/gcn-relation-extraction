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

        self.nlp = nlp
        self.sentences, self.labels, self.adj_matrices = self._get_data(path)

        # only initialize vocab for training set
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = build_vocab_from_iterator(
                self.tokens, specials=[self.pad, self.unk]
            )
            self.vocab.set_default_index(self.vocab[self.unk])

    def _get_data(self, path):
        sentences: list[list[str]] = []
        labels: list[str] = []
        adj_matrices: list[torch.Tensor] = []

        with open(path, 'r', encoding='utf8') as f:
            for line in DictReader(f, fieldnames=self.columns, delimiter="\t"):
                # to skip all examples with class "Other", since it's common
                # if label == "Other":
                #     continue

                doc = self.nlp(line["sentence"])

                sentence = [token.text for token in doc]
                label = line["relation"]
                adj_matrix = deps.generate_matrix(doc)

                sentences.append(sentence)
                labels.append(label)
                adj_matrices.append(adj_matrix)

        return sentences, labels, adj_matrices

    @property
    def tokens(self):
        return iter(self.sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        x = torch.tensor(self.vocab(self.sentences[idx]))
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
