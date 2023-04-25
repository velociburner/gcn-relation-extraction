import re
import torch

from spacy.tokens import Doc, Token


class WhitespaceTokenizer:
    """Custom spaCy tokenizer that just tokenizes periods and splits on
    spaces."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        # strip surrounding whitespace and tokenize sentence final period
        text = re.sub(r"(.+)\.$", r"\1 .", text.strip())
        words = text.split(" ")
        spaces = [True] * len(words)
        spaces[-1] = False
        return Doc(self.vocab, words=words, spaces=spaces)


def generate_matrix(doc: Doc) -> torch.Tensor:
    """Generates adjacency matrix given a document."""
    # each example is only a single sentence
    sentence = next(doc.sents)
    root = sentence.root
    adj_matrix = torch.zeros((len(sentence), len(sentence)))
    _tree_to_matrix(root, adj_matrix)

    # root node is included in its children, so remove that edge
    adj_matrix[root.i, root.i] = 0

    return adj_matrix


def _tree_to_matrix(node: Token, adj_matrix: torch.Tensor):
    """Traverses a dependency tree and adds a 1 to the adjacency matrix in each
    position corresponding to a dependency arc."""
    adj_matrix[node.head.i, node.i] = 1
    if node.n_lefts > 0:
        for child in node.lefts:
            _tree_to_matrix(child, adj_matrix)
    if node.n_rights > 0:
        for child in node.rights:
            _tree_to_matrix(child, adj_matrix)
