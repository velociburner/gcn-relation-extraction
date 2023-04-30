import click
import matplotlib.pyplot as plt
import numpy as np
import spacy
import torch
import torch.nn as nn

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from torch.utils.data import DataLoader, Subset
from torchtext.vocab import GloVe, pretrained_aliases

import deps
from dataset import CollateSemEval, SemEvalDataset
from model import GCNClassifier


def get_parser(model_name="en_core_web_sm"):
    """Loads a dependency parser from spaCy."""
    nlp = spacy.load(model_name)

    # add custom tokenizer and sentence segmentation based on dataset format
    nlp.tokenizer = deps.WhitespaceTokenizer(nlp.vocab)
    nlp.add_pipe("simple_sentencizer", before="parser")

    # disable everything not needed for parsing
    nlp.disable_pipes(['tagger', 'attribute_ruler', 'lemmatizer', 'ner'])

    return nlp


def get_data_loaders(train_data, dev_data, test_data, batch_size, collate_fn):
    """Creates the DataLoader objects for each data split."""
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    return train_loader, dev_loader, test_loader


def train(model, train_loader, dev_loader, epochs, lr, clip):
    """Trains a model on the training data."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("Epoch\tLoss\tDev acc")
    for epoch in range(epochs):
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            sentences, labels, e1s, e2s, adj_matrices = data
            optimizer.zero_grad()

            output = model(sentences, adj_matrices, e1s, e2s)
            loss = loss_fn(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()

            epoch_loss += loss.item() / len(sentences)

        preds, golds = predict(model, dev_loader)
        acc = accuracy_score(golds, preds)
        print(f"{epoch + 1}\t{epoch_loss:.2f}\t{acc:.4f}")


def predict(model, loader):
    """Gets model predictions for unseen data."""
    preds_list: list[torch.Tensor] = []
    golds_list: list[torch.Tensor] = []

    with torch.no_grad():
        for sentences, labels, e1s, e2s, adj_matrices in loader:
            output = model(sentences, adj_matrices, e1s, e2s)
            preds = torch.argmax(output, dim=-1)
            preds_list.append(preds.cpu().numpy())
            golds_list.append(labels.cpu().numpy())

    preds: np.ndarray = np.concatenate(preds_list)
    golds: np.ndarray = np.concatenate(golds_list)

    return preds, golds


def display_results(preds: np.ndarray, golds: np.ndarray):
    """Prints the classification metrics and displays the confusion matrix
    comparing the predictions and gold labels."""
    print(classification_report(golds, preds, digits=4, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(golds, preds)
    plt.show()


@click.command()
@click.option('--epochs', type=int, default=10,
              help="Number of training epochs")
@click.option('--dr', type=float, default=0.5, help="Dropout value")
@click.option('--lr', type=float, default=5e-4, help="Learning rate")
@click.option('--batch-size', type=int, default=64, help="Batch size")
@click.option('--clip', type=int, default=10, help="Gradient clip value")
@click.option('--patience', type=int, default=5,
              help="Number of epochs to wait before early stopping")
@click.option('--embed-dim', type=click.Choice(['50', '100', '200', '300']),
              default='50', help="Dimensionality of embeddings")
@click.option('--use-pretrained', is_flag=True,
              help="Use pretrained word embeddings")
@click.option('--use_lstm', is_flag=True, help="Use LSTM encoder")
@click.option('--bidirectional', is_flag=True, help="Use BiLSTM")
@click.option('--tune', is_flag=True,
              help="Perform a grid search over all models and hyperparameters")
@click.option('--num-folds', type=int, default=10,
              help="Number of folds to use for K-fold cross-validation")
@click.option('--device', type=click.Choice(['cpu', 'cuda', 'mps']),
              default='cpu', help="Device to run on")
def main(**kwargs):
    nlp = get_parser()

    print("Loading train data...")
    train_data = SemEvalDataset(nlp)
    vocab = train_data.vocab

    print("Loading test data...")
    test_data = SemEvalDataset(nlp, split="test", vocab=vocab)

    # split into train and dev
    train_split = Subset(train_data, list(range(7000)))
    dev_split = Subset(train_data, list(range(7000, 8000)))

    # data loaders
    collate_fn = CollateSemEval(kwargs["device"])
    train_loader, dev_loader, test_loader = get_data_loaders(
        train_split, dev_split, test_data, kwargs["batch_size"], collate_fn
    )

    embed_dim = int(kwargs["embed_dim"])
    if kwargs["use_pretrained"]:
        embedding_source = f"glove.6B.{embed_dim}d"
        print(f"Loading embeddings from {embedding_source}...")
        pretrained: GloVe = pretrained_aliases[embedding_source]()
        print("Loaded")
    else:
        pretrained = None

    num_classes = len(set(train_data.labels))
    model = GCNClassifier(
        len(vocab),
        embed_dim,
        num_classes=num_classes,
        dr=kwargs["dr"],
        use_lstm=kwargs["use_lstm"],
        pretrained=pretrained,
        vocab=vocab
    )

    model = model.to(kwargs["device"])

    epochs, lr, clip = kwargs["epochs"], kwargs["lr"], kwargs["clip"]
    print("Training...")
    train(model, train_loader, dev_loader, epochs, lr, clip)
    print()
    print("Predicting on test set...")
    preds, golds = predict(model, test_loader)
    display_results(preds, golds)


if __name__ == "__main__":
    main()
