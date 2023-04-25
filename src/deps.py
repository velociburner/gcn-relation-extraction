import torch

from spacy.language import Language
from spacy.tokens import Token


def generate_matrices(examples: list[str], nlp: Language) -> list[torch.Tensor]:
    """Generates adjacency matrices given a list of examples."""
    adj_matrices: list[torch.Tensor] = []

    for example in examples:
        # disable everything not needed for parsing
        doc = nlp(example, disable=['tagger', 'attribute_ruler', 'lemmatizer', 'ner'])

        # each example is only a single sentence
        sentence = next(doc.sents)
        root = sentence.root
        adj_matrix = torch.zeros((len(sentence), len(sentence)))
        tree_to_matrix(root, adj_matrix)

        # root node is included in its children, so remove that edge
        adj_matrix[root.i, root.i] = 0
        adj_matrices.append(adj_matrix)

    return adj_matrices


def tree_to_matrix(node: Token, adj_matrix: torch.Tensor):
    """Traverses a dependency tree and adds a 1 to the adjacency matrix in each
    position corresponding to a dependency arc."""
    adj_matrix[node.head.i, node.i] = 1
    if node.n_lefts > 0:
        for child in node.lefts:
            tree_to_matrix(child, adj_matrix)
    if node.n_rights > 0:
        for child in node.rights:
            tree_to_matrix(child, adj_matrix)
