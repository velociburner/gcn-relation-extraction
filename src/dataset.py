import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

import deps


@dataclass
class SemEvalExample:
    sentence: torch.Tensor
    e1: tuple[int, int]
    e2: tuple[int, int]
    adj_matrix: torch.Tensor
    label: int


class SemEvalDataset(Dataset):

    def __init__(self, nlp, split="train", vocab=None):
        self.name = "sem_eval_2010_task_8"
        self.nlp = nlp
        self.split = split
        self.sentences, self.entities, self.adj_matrices, self.labels = self._get_data()
        self.num_classes = len(set(self.labels))

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

    def _get_data(self):
        """Loads the sentences and labels from the SemEval dataset and
        calculates the entity indices and adjacency matrices for each
        example."""
        data = load_dataset(self.name, split=self.split)

        sentences: list[list[str]] = []
        labels: list[int] = data['relation']
        entities: list[tuple[int, int]] = []
        adj_matrices: list[torch.Tensor] = []

        # span surrounded by an opening and closing entity tag
        pattern = re.compile(r'<e\d>(.*?)<\/e\d>')

        for sentence in tqdm(data['sentence']):
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

        return sentences, entities, adj_matrices, labels

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

    def __getitem__(self, index):
        sentence = torch.tensor(self.vocab(self.sentences[index]))
        e1 = self.entities[index][0]
        e2 = self.entities[index][1]
        adj_matrix = self.adj_matrices[index]
        label = self.labels[index]
        return SemEvalExample(sentence, e1, e2, adj_matrix, label)


class CollateSemEval:
    """Class used as collate_fn for PyTorch Dataloaders. It simply pads each
    sequence to the longest sequence length in the batch and returns it, along
    with a tensor of labels and padded adjacency matrix of dependency arcs."""

    def __init__(self, padding_value: float = 0.):
        self.padding_value = padding_value

    def __call__(self, examples: list[SemEvalExample]):
        # if the input includes explicit labels (by sklearn), discard them
        if isinstance(examples[0], tuple):
            examples = [example[0] for example in examples]

        sentences = nn.utils.rnn.pad_sequence(
            [example.sentence for example in examples],
            batch_first=True,
            padding_value=self.padding_value
        )
        e1s = torch.tensor([example.e1 for example in examples])
        e2s = torch.tensor([example.e2 for example in examples])
        adj_matrices = [example.adj_matrix for example in examples]
        labels = torch.tensor([example.label for example in examples])

        # pad each adjacency matrix to the size of the largest one
        # each matrix is padded along both dimensions and remains a square matrix
        max_size = max([matrix.size()[0] for matrix in adj_matrices])
        padded_matrices = torch.stack([
            F.pad(
                matrix,
                (0, max_size - matrix.size()[1], 0, max_size - matrix.size()[0]),
                value=self.padding_value
            )
            for matrix in adj_matrices
        ])

        inputs = sentences, e1s, e2s, padded_matrices
        return inputs, labels
