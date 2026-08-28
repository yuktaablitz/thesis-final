"""
Stage 12B — Temporal Convolutional Network (Bai, Kolter & Koltun, 2018
style: stacked dilated causal 1D convolutions with residual connections).
Dilation doubles per layer (1, 2, 4, 8, ...) so a modest depth covers a
full 72h/168h window without an LSTM's sequential bottleneck.

Runs on CPU, Apple Silicon (torch.device("mps")), or CUDA (Colab)
unchanged — device is just a constructor arg to BtcTCN.forward's caller.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """Removes the trailing padding a causal conv adds, so output length == input length."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int,
                 dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal: pad left only (via Chomp1d on the right)

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                            padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.drop1,
                                  self.conv2, self.chomp2, self.relu2, self.drop2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, n_features: int, channels: list[int], kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(channels):
            in_channels = n_features if i == 0 else channels[i - 1]
            dilation = 2 ** i
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) -> (batch, seq_len, channels[-1])."""
        x = x.transpose(1, 2)          # -> (batch, n_features, seq_len)
        out = self.network(x)
        return out.transpose(1, 2)     # -> (batch, seq_len, channels[-1])


class BtcTCN(nn.Module):
    """Full classifier: TCN over the window, take the last timestep, sigmoid head."""

    def __init__(self, n_features: int, channels: list[int] | None = None,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        channels = channels or [64, 64, 64, 64]
        self.tcn = TemporalConvNet(n_features, channels, kernel_size, dropout)
        self.head = nn.Sequential(nn.Linear(channels[-1], 32), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_hours, n_features) -> (batch,) logit for P(BTC UP)."""
        features = self.tcn(x)
        last_step = features[:, -1, :]
        return self.head(last_step).squeeze(-1)
