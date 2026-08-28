"""
Stage 12C — CNN-LSTM. 1D conv layers find local temporal patterns over the
window; an LSTM then learns the sequential relationship between those
patterns before a sigmoid head.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BtcCNNLSTM(nn.Module):
    def __init__(self, n_features: int, conv_channels: int = 64, lstm_hidden: int = 64,
                 lstm_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, num_layers=lstm_layers,
                             batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(lstm_hidden, 32), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_hours, n_features) -> (batch,) logit for P(BTC UP)."""
        x = x.transpose(1, 2)              # -> (batch, n_features, seq_len)
        local_patterns = self.conv(x)
        local_patterns = local_patterns.transpose(1, 2)  # -> (batch, seq_len, conv_channels)
        _, (h_n, _) = self.lstm(local_patterns)
        last_hidden = h_n[-1]              # (batch, lstm_hidden)
        return self.head(last_hidden).squeeze(-1)
