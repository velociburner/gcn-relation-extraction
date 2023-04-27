import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

import deps


@dataclass
class SemEvalExample:
    sentence: torch.Tensor
    label: int
    e1: tuple[int, int]
    e2: tuple[int, int]
    adj_matrix: torch.Tensor


class SemEvalDataset(Dataset):

    def __init__(self, nlp, start, end, split="train", vocab=None):
        self.name = "sem_eval_2010_task_8"
        self.nlp = nlp
        self.split = split
        self.sentences, self.labels, self.entities, self.adj_matrices = self._get_data(start, end)

        self.pad = "<pad>"
        self.unk = "<unk>"

        # only initialize vocab for training set
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = build_vocab_from_iterator(
                self.tokens, specials=[self.pad, self.unk]
            )
            self.vocab.set_default_index(self.vocab[self.unk])

    def _get_data(self, start: int, end: int):
        """Loads the sentences and labels from the SemEval dataset and
        calculates the entity indices and adjacency matrices for each
        example."""
        data = load_dataset(self.name, split=self.split)[start: end]

        sentences: list[list[str]] = []
        labels: list[int] = data['relation']
        entities: list[tuple[int, int]] = []
        adj_matrices: list[torch.Tensor] = []

        # span surrounded by an opening and closing entity tag
        pattern = re.compile(r'<e\d>(.*?)<\/e\d>')

        for sentence in data['sentence']:
            # filter out entity tags
            matches = re.findall(pattern, sentence)
            clean_text = re.sub(pattern, r' \1 ', sentence)

            doc = self.nlp(clean_text)

            tokens: list[str] = [token.text for token in doc]
            entity_idxs = self._get_entity_idxs(tokens, matches)
            adj_matrix = deps.generate_matrix(doc)

            sentences.append(tokens)
            entities.append(entity_idxs)
            adj_matrices.append(adj_matrix)

        return sentences, labels, entities, adj_matrices

    def _get_entity_idxs(self, tokens: list[str], matches: list[Any]):
        """Calculates the start and end indices of each entity in the list of
        matches."""
        entity_idxs: list[tuple[int, int]] = []

        for match in matches:
            entity_tokens = match.split()
            start_idx = -1

            for i, token in enumerate(tokens):
                if tokens[i: i + len(entity_tokens)] == entity_tokens:
                    start_idx = i
                    break

            # make sure we found the entity span
            assert start_idx >= 0
            end_idx = start_idx + len(entity_tokens) - 1
            entity_idxs.append((start_idx, end_idx))

        return entity_idxs

    @property
    def tokens(self):
        return iter(self.sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = torch.tensor(self.vocab(self.sentences[idx]))
        label = self.labels[idx]
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
