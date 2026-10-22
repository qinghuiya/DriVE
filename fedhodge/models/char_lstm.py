from __future__ import annotations

import torch
from torch import nn


class CharLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        output, _ = self.lstm(emb)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)
        return logits
