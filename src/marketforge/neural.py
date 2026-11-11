"""Optional CPU neural estimators for engineered tabular market features.

The attention model attends across feature tokens in one observation. It is not
a temporal Transformer and never treats adjacent panel rows as a sequence.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch import nn


class _GatedBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.value = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )
        self.gate = nn.Linear(width, width)

    def forward(self, x):
        normalized = self.norm(x)
        return x + torch.sigmoid(self.gate(normalized)) * self.value(normalized)


class _ResidualNetwork(nn.Module):
    def __init__(self, n_features: int, width: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, width), nn.GELU(),
            _GatedBlock(width, dropout), _GatedBlock(width, dropout),
            nn.LayerNorm(width), nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class _FeatureAttentionNetwork(nn.Module):
    def __init__(self, n_features: int, width: int, dropout: float):
        super().__init__()
        self.feature_weight = nn.Parameter(torch.randn(n_features, width) * 0.05)
        self.feature_bias = nn.Parameter(torch.randn(n_features, width) * 0.05)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=width, nhead=4, dim_feedforward=width * 2,
                dropout=dropout, batch_first=True, activation="gelu",
                norm_first=True,
            )
            for _ in range(2)
        ])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.feature_weight + self.feature_bias
        encoded = torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1)
        for block in self.blocks:
            encoded = block(encoded)
        return self.output(encoded[:, 0]).squeeze(-1)


class TorchTabularRegressor(RegressorMixin, BaseEstimator):
    """Deterministic-seeded CPU fitting with training-only standardization.

    Uses fixed epoch budgets, AdamW, Huber loss and gradient clipping. No random
    validation split or test-driven stopping is used. Parameter selection belongs
    to the external purged time-series experiment. ``loss_history_`` records
    training losses for audit; it is not a generalization estimate.
    """

    def __init__(
        self,
        architecture: str = "attention",
        width: int = 32,
        epochs: int = 30,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        dropout: float = 0.1,
        random_state: int = 42,
    ):
        self.architecture = architecture
        self.width = width
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        if self.architecture not in {"attention", "residual"}:
            raise ValueError("architecture must be attention or residual")
        if self.width < 4 or (self.architecture == "attention" and self.width % 4):
            raise ValueError("width must be >=4 and divisible by 4 for attention")
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")
        if self.weight_decay < 0 or not 0 <= self.dropout < 1:
            raise ValueError("weight_decay must be nonnegative and dropout in [0,1)")
        self.n_features_in_ = X.shape[1]
        self.scaler_ = StandardScaler().fit(X)
        self.target_mean_ = float(np.mean(y))
        self.target_scale_ = max(float(np.std(y)), 1e-8)
        self.constant_target_ = bool(np.ptp(y) < 1e-12)
        self.loss_history_ = []
        features = torch.from_numpy(self.scaler_.transform(X).astype(np.float32))
        target = torch.from_numpy(((y - self.target_mean_) / self.target_scale_).astype(np.float32))
        # Restore the caller's random state afterwards; CPU is intentional.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            network = _FeatureAttentionNetwork if self.architecture == "attention" else _ResidualNetwork
            self.network_ = network(X.shape[1], self.width, self.dropout).cpu()
            if not self.constant_target_:
                optimizer = torch.optim.AdamW(
                    self.network_.parameters(), lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
                loss_function = nn.HuberLoss(delta=1.0)
                self.network_.train()
                for _ in range(self.epochs):
                    order = torch.randperm(len(features))
                    total = 0.0
                    for start in range(0, len(features), self.batch_size):
                        selection = order[start : start + self.batch_size]
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_function(self.network_(features[selection]), target[selection])
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Neural training produced a nonfinite loss")
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.network_.parameters(), max_norm=1.0)
                        optimizer.step()
                        total += float(loss.detach()) * len(selection)
                    self.loss_history_.append(total / len(features))
            self.network_.eval()
        return self

    def predict(self, X):
        check_is_fitted(self, "network_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count differs from training")
        if self.constant_target_:
            return np.full(len(X), self.target_mean_)
        features = torch.from_numpy(self.scaler_.transform(X).astype(np.float32))
        self.network_.eval()
        with torch.no_grad():
            output = torch.cat([
                self.network_(features[start : start + self.batch_size])
                for start in range(0, len(features), self.batch_size)
            ]).numpy()
        predictions = output.astype(float) * self.target_scale_ + self.target_mean_
        if not np.isfinite(predictions).all():
            raise FloatingPointError("Neural inference produced nonfinite predictions")
        return predictions
