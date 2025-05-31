import torch
import torch.nn as nn

class Algorithm(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim, pad_idx, n_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim * 2, out_dim)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(self.dropout(h))