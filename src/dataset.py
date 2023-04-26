from csv import DictReader
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

import deps


@dataclass
class SemEvalExample:
    sentence: torch.Tensor
    label: torch.Tensor
    e1: int
    e2: int
    adj_matrix: torch.Tensor


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
        self.sentences, self.labels, self.entities, self.adj_matrices = self._get_data(path)

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
        entities: list[tuple[int, int]] = []
        adj_matrices: list[torch.Tensor] = []

        with open(path, 'r', encoding='utf8') as f:
            for line in DictReader(f, fieldnames=self.columns, delimiter="\t"):
                # to skip all examples with class "Other", since it's common
                # if label == "Other":
                #     continue

                doc = self.nlp(line["sentence"])

                sentence = [token.text for token in doc]
                label = line["relation"]
                entity = int(line["e1"]), int(line["e2"])
                adj_matrix = deps.generate_matrix(doc)

                sentences.append(sentence)
                labels.append(label)
                entities.append(entity)
                adj_matrices.append(adj_matrix)

        return sentences, labels, entities, adj_matrices

    @property
    def tokens(self):
        return iter(self.sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = torch.tensor(self.vocab(self.sentences[idx]))
        label = torch.tensor(self.label_map[self.labels[idx]])
        e1 = self.entities[idx][0]
        e2 = self.entities[idx][1]
        adj_matrix = self.adj_matrices[idx]
        return SemEvalExample(sentence, label, e1, e2, adj_matrix)


def collate_fn(examples: Iterable[SemEvalExample], padding_value: float = 0.):
    """Function used as collate_fn for PyTorch Dataloaders. It simply pads each
    sequence to the longest sequence length in the batch and returns it, along
    with a tensor of labels and padded adjacency matrix of dependency arcs."""
    sentences = nn.utils.rnn.pad_sequence(
        [example.sentence for example in examples],
        batch_first=True,
        padding_value=padding_value
    )
    labels = torch.tensor([example.label for example in examples])
    e1s = torch.tensor([example.e1 for example in examples])
    e2s = torch.tensor([example.e2 for example in examples])
    adj_matrices = [example.adj_matrix for example in examples]

    # pad each adjacency matrix to the size of the largest one
    # each matrix is padded along both dimensions and remains a square matrix
    max_size = max([matrix.size()[0] for matrix in adj_matrices])
    padded_matrices = torch.stack([
        F.pad(
            matrix,
            (0, max_size - matrix.size()[1], 0, max_size - matrix.size()[0]),
            value=padding_value
        )
        for matrix in adj_matrices
    ])

    return sentences, labels, e1s, e2s, padded_matrices
