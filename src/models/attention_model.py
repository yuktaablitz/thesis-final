"""
Approach 2, Stage 7-11 — cross-market attention model. Each modality (BTC,
derivatives, cross-market, news, macro) gets its own small TCN encoder
("branch"); temporal multi-head self-attention then lets each branch decide
which historical hours matter; modality attention finally lets the model
learn a per-prediction weighting over branches instead of a flat concat.

Build and compare the three variants named in configs/approach2.yaml
`compare_variants` (Stage 11) — toggle temporal_attention / modality_attention
below to reproduce each one:
    model_a_concat            : both False  (TCN, concatenate branch outputs)
    model_b_temporal_attention: temporal=True, modality=False
    model_c_full_attention    : both True   (the full architecture)

modality_weights are returned alongside the prediction so you can inspect,
per sample, which branch the model leaned on (e.g. news spiking during an
ETF announcement) — see design doc Stage 10 example.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.tcn import TemporalConvNet


class TemporalEncoder(nn.Module):
    """TCN over one modality's window, optionally followed by self-attention over time."""

    def __init__(self, n_features: int, channels: list[int], n_heads: int = 4,
                 use_temporal_attention: bool = True, dropout: float = 0.2):
        super().__init__()
        self.tcn = TemporalConvNet(n_features, channels, dropout=dropout)
        self.use_temporal_attention = use_temporal_attention
        if use_temporal_attention:
            self.attn = nn.MultiheadAttention(channels[-1], n_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(channels[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) -> (batch, channels[-1]) pooled representation."""
        h = self.tcn(x)  # (batch, seq_len, channels[-1])
        if self.use_temporal_attention:
            attended, _ = self.attn(h, h, h, need_weights=False)
            h = self.norm(h + attended)
        return h[:, -1, :]  # last timestep = representation as of prediction_time


class ModalityAttention(nn.Module):
    """
    Learns a query vector; attends over the stacked per-modality embeddings
    to produce one fused representation plus interpretable per-modality
    weights (Stage 10).
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

    def forward(self, modality_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """modality_embeddings: (batch, n_modalities, dim) -> (fused: (batch, dim), weights: (batch, n_modalities))."""
        batch = modality_embeddings.shape[0]
        query = self.query.expand(batch, -1, -1)
        fused, weights = self.attn(query, modality_embeddings, modality_embeddings, need_weights=True,
                                    average_attn_weights=True)
        return fused.squeeze(1), weights.squeeze(1)


class BtcCrossMarketAttentionModel(nn.Module):
    """
    modality_input_dims: e.g. {"btc": 40, "derivatives": 12, "cross_market": 24,
    "news": 15, "macro": 6} — feature count per modality's window tensor.
    Modalities missing from a given batch (e.g. news unavailable that hour)
    should be zero-filled upstream; this model doesn't handle ragged inputs.
    """

    def __init__(self, modality_input_dims: dict[str, int], embed_dim: int = 64,
                 n_heads: int = 4, use_temporal_attention: bool = True,
                 use_modality_attention: bool = True, dropout: float = 0.2):
        super().__init__()
        self.modalities = list(modality_input_dims.keys())
        self.use_modality_attention = use_modality_attention

        self.encoders = nn.ModuleDict({
            name: TemporalEncoder(dim, channels=[embed_dim, embed_dim], n_heads=n_heads,
                                   use_temporal_attention=use_temporal_attention, dropout=dropout)
            for name, dim in modality_input_dims.items()
        })

        if use_modality_attention:
            self.modality_attn = ModalityAttention(embed_dim, n_heads, dropout)
            head_in = embed_dim
        else:
            head_in = embed_dim * len(self.modalities)  # plain concatenation (Model A)

        self.head = nn.Sequential(nn.Linear(head_in, 32), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(32, 1))

    def forward(self, modality_windows: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        modality_windows: {name: (batch, seq_len, n_features)} for each key in self.modalities.
        Returns (logits: (batch,), modality_weights: (batch, n_modalities) or None for Model A).
        """
        embeddings = [self.encoders[name](modality_windows[name]) for name in self.modalities]

        if self.use_modality_attention:
            stacked = torch.stack(embeddings, dim=1)  # (batch, n_modalities, embed_dim)
            fused, weights = self.modality_attn(stacked)
        else:
            fused = torch.cat(embeddings, dim=1)
            weights = None

        logits = self.head(fused).squeeze(-1)
        return logits, weights
