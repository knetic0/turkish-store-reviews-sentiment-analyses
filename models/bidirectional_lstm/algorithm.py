import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class Algorithm(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim, pad_idx, n_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers>1 else 0.0
        )
        self.fc   = nn.Linear(hid_dim * 2, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, texts, lengths):
        emb     = self.drop(self.embedding(texts))
        packed  = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        h_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.drop(h_cat))