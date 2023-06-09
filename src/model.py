import torch
from torch import nn
from torchtext.vocab import GloVe, Vocab


def from_pretrained(pretrained: GloVe, vocab: Vocab):
    """Uses pretrained GloVe word embeddings to generate an embedding
    matrix with special tokens included."""
    unk = torch.mean(pretrained.vectors, dim=0)
    pretrained = pretrained.get_vecs_by_tokens(vocab.get_itos())
    pretrained[vocab["<unk>"]] = unk
    return nn.Embedding.from_pretrained(pretrained)


class GCNLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, layer_inputs, adj_matrices):
        """Runs the previous hidden states through the corresponding linear
        layer, applies a graph convolution using the adjacency matrices, and
        adds that to the hidden outputs for self loops, all weighted by the
        degree of each graph node."""

        # take the sum of each token's connections and add 1 for self loops
        # (batch_size, seq_len, seq_len) -> (batch_size, seq_len, 1)
        token_degrees = adj_matrices.sum(-1).unsqueeze(-1) + 1

        # (batch_size, seq_len, gcn_hidden_dim)
        hidden_outputs = self.layer(layer_inputs)
        convolution = torch.bmm(adj_matrices, hidden_outputs)
        layer_outputs = (convolution + hidden_outputs) / token_degrees

        return layer_outputs


class GCNClassifier(nn.Module):

    def __init__(
        self,
        emb_input_dim,
        emb_output_dim,
        gcn_hidden_dim=100,
        num_layers=3,
        num_classes=2,
        dr=0.5,
        pretrained=None,
        vocab=None,
        use_lstm=True,
        lstm_hidden_dim=100,
        lstm_layers=2,
        bidirectional=True
    ):
        super().__init__()
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)

        self.dr = nn.Dropout(dr)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

        if use_lstm:
            self.lstm = nn.LSTM(
                emb_output_dim,
                lstm_hidden_dim,
                batch_first=True,
                num_layers=lstm_layers,
                dropout=dr,
                bidirectional=bidirectional
            )
        self.use_lstm = use_lstm
        self.lstm_layers = lstm_layers
        # BiLSTM concatenates two outputs
        if bidirectional:
            lstm_hidden_dim *= 2

        self.gcn_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                gcn_input_dim = lstm_hidden_dim if use_lstm else emb_output_dim
            else:
                gcn_input_dim = gcn_hidden_dim
            self.gcn_layers.append(GCNLayer(gcn_input_dim, gcn_hidden_dim))

        # output of GCN is 3 concatenated hidden vectors
        self.project = nn.Linear(gcn_hidden_dim * 3, num_classes)
        self.apply(self._init_weights)

        # load pretrained embeddings after initializing weights
        if pretrained is not None:
            self.embedding = from_pretrained(pretrained, vocab)

    def forward(self, inputs):
        """Obtains embeddings, runs through an LSTM, the GCN layers, and a
        final projection layer.

        See https://aclanthology.org/D18-1244.pdf for details (pp. 2206-2208).
        """
        sentences, e1s, e2s, adj_matrices = inputs

        # (batch_size, seq_len, embed_dim)
        embedding = self.embedding(sentences)

        if self.use_lstm:
            gcn_input, _ = self.lstm(embedding)
        else:
            gcn_input = embedding

        for i, gcn_layer in enumerate(self.gcn_layers):
            gcn_input = gcn_layer(gcn_input, adj_matrices)
            gcn_input = self.relu(gcn_input)
            if i < len(self.gcn_layers) - 1:
                gcn_input = self.dr(gcn_input)

        hidden_final = self.pool_cat(gcn_input, e1s, e2s)

        # (batch_size, num_classes)
        logits = self.project(hidden_final)
        return logits

    def pool_cat(self, gcn_hidden, e1s, e2s):
        """Pools the hidden vectors for each entity and concatenates them
        with the pooled hidden representation for the entire sentence."""
        gcn_hidden_dim = gcn_hidden.size()[-1]

        # swap sequence and hidden dimensions to pool over final dimension
        # (batch_size, seq_len, gcn_hidden_dim) -> (batch_size, gcn_hidden_dim, seq_len)
        gcn_hidden = gcn_hidden.transpose(1, 2)

        # (batch_size, 2) -> (batch_size, 1, 2)
        e1s = e1s.unsqueeze(1)
        e2s = e2s.unsqueeze(1)

        # (batch_size, 1, 2) -> (batch_size, gcn_hidden_dim, 2)
        e1_idxs = e1s.expand(-1, gcn_hidden_dim, -1)
        e2_idxs = e2s.expand(-1, gcn_hidden_dim, -1)

        # start and end vectors for each entity span
        # (batch_size, gcn_hidden_dim, 2)
        hidden_subj = torch.gather(gcn_hidden, -1, e1_idxs)
        hidden_obj = torch.gather(gcn_hidden, -1, e2_idxs)

        # pool over sequence dimension
        # (batch_size, gcn_hidden_dim)
        hidden_subj = self.pool(hidden_subj).squeeze(-1)
        hidden_obj = self.pool(hidden_obj).squeeze(-1)
        hidden_sent = self.pool(gcn_hidden).squeeze(-1)

        # (batch_size, gcn_hidden_dim * 3)
        hidden_final = torch.cat([hidden_subj, hidden_sent, hidden_obj], dim=1)

        return hidden_final

    def _init_weights(self, module):
        """Initializes the weights of each layer of the module using Xavier
        initialization. Also sets each bias to 0."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for i in range(self.lstm_layers):
                nn.init.xavier_uniform_(module.all_weights[i][0])
                nn.init.xavier_uniform_(module.all_weights[i][1])
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
