from csv import DictReader
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator


class SemEvalDataset(Dataset):

    def __init__(self, path, labels, vocab=None):
        self.columns = ["relation", "e1", "e2", "sentence"]
        self.label_map = {label: i for i, label in enumerate(labels)}
        self.pad = "<pad>"
        self.unk = "<unk>"

        self.e1_start = "<e1>"
        self.e1_end = "</e1>"
        self.e2_start = "<e2>"
        self.e2_end = "</e2>"

        self.sentences, self.labels = self._get_data(path)

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
        labels: list[str] = []
        with open(path, 'r', encoding='utf8') as f:
            for line in DictReader(f, fieldnames=self.columns, delimiter="\t"):
                # sentences with entity at index 0 start with a space
                sentence = line["sentence"].strip().split(" ")
                label = line["relation"]

                # to skip all examples with class "Other", since it's common
                # if label == "Other":
                #     continue

                e1 = int(line["e1"])
                e2 = int(line["e2"])

                # surround both entities with special tokens
                sentence.insert(e1, self.e1_start)
                sentence.insert(e1 + 2, self.e1_end)
                sentence.insert(e2 + 2, self.e2_start)
                sentence.insert(e2 + 4, self.e2_end)

                sentences.append(sentence)
                labels.append(label)

        return sentences, labels

    @property
    def tokens(self):
        for sentence in self.sentences:
            yield sentence

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        x = torch.tensor(self.vocab(self.sentences[index]))
        y = torch.tensor(self.label_map[self.labels[index]])
        return x, y


def collate_fn(examples: Iterable[tuple], padding_value: float = 0.):
    """Function used as collate_fn for PyTorch Dataloaders. It simply pads each
    sequence to the longest sequence length in the batch and returns it, along
    with a tensor of labels."""
    token_indices = nn.utils.rnn.pad_sequence(
        [example[0] for example in examples],
        batch_first=True,
        padding_value=padding_value
    )
    labels = torch.tensor([example[1] for example in examples])

    return token_indices, labels
