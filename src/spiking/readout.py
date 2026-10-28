"""Readout classifier wrapper for reservoir models (pure NumPy closed-form Ridge + optional sklearn)."""

from __future__ import annotations

import time
import numpy as np

from src.spiking.lif_reservoir import ReservoirOutput, SpikingLIFReservoir


class BalancedRidgeReadout:
    """Closed-form class-balanced L2-regularized linear readout (identical objective to sklearn RidgeClassifier)."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = float(alpha)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None       # (D, K)
        self.intercept_: np.ndarray | None = None  # (1, K)
        self.n_classes_: int = 6

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BalancedRidgeReadout":
        N, D = X.shape
        classes = np.unique(y)
        K = len(classes)
        self.n_classes_ = K

        # Standardize features
        self.mean_ = X.mean(axis=0, keepdims=True).astype(np.float32)
        self.std_ = (X.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        Xs = ((X - self.mean_) / self.std_).astype(np.float64)

        # Append bias column
        X_aug = np.hstack([Xs, np.ones((N, 1), dtype=np.float64)])

        # One-hot targets {-1, +1}
        Y = np.full((N, K), -1.0, dtype=np.float64)
        for idx, c in enumerate(classes):
            Y[y == c, idx] = 1.0

        # Class-balanced sample weights: N / (K * count(c))
        counts = np.bincount(y, minlength=K).astype(np.float64)
        class_weights = N / (K * np.maximum(counts, 1.0))
        sample_weights = class_weights[y]  # (N,)

        # Weighted normal equations: (X^T W X + alpha * I) W_out = X^T W Y
        WX = X_aug * sample_weights[:, None]
        A = X_aug.T @ WX
        reg = np.eye(D + 1, dtype=np.float64) * self.alpha
        reg[-1, -1] = 0.0  # Do not regularize intercept
        B_mat = WX.T @ Y

        W_full = np.linalg.solve(A + reg, B_mat).astype(np.float32)
        self.coef_ = W_full[:-1, :]
        self.intercept_ = W_full[-1:, :]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        assert self.coef_ is not None and self.intercept_ is not None
        Xs = (X - self.mean_) / self.std_
        scores = Xs @ self.coef_ + self.intercept_
        return np.argmax(scores, axis=1).astype(np.int64)


class SpikingReservoirClassifier:
    """End-to-end Spiking Reservoir + Linear Readout classifier."""

    def __init__(
        self,
        reservoir: SpikingLIFReservoir,
        readout_alpha: float = 10.0,
    ) -> None:
        self.reservoir = reservoir
        self.readout_alpha = float(readout_alpha)
        self.clf = BalancedRidgeReadout(alpha=self.readout_alpha)
        self.train_time_sec: float = 0.0
        self.train_reservoir_output: ReservoirOutput | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SpikingReservoirClassifier":
        """Extract reservoir features on clean training data and train only the linear readout."""
        t0 = time.perf_counter()
        res_out = self.reservoir.run_numpy(X_train, return_traces=False)
        self.clf.fit(res_out.features, y_train)
        self.train_time_sec = time.perf_counter() - t0
        self.train_reservoir_output = res_out
        return self

    def predict_and_measure(
        self, X_eval: np.ndarray, return_traces: bool = False
    ) -> tuple[np.ndarray, ReservoirOutput, float]:
        """Run reservoir on evaluation windows and return predictions, reservoir stats, and latency."""
        t0 = time.perf_counter()
        res_out = self.reservoir.run_numpy(X_eval, return_traces=return_traces)
        preds = self.clf.predict(res_out.features)
        total_eval_sec = time.perf_counter() - t0
        return preds, res_out, total_eval_sec

    @property
    def model_size_bytes(self) -> int:
        """Compute total memory footprint of reservoir + readout parameters in bytes."""
        res_bytes = (
            self.reservoir.W_in.nbytes
            + self.reservoir.W_rec.nbytes
            + self.reservoir.tau_m.nbytes
            + self.reservoir.bias.nbytes
        )
        assert self.clf.coef_ is not None and self.clf.intercept_ is not None
        readout_bytes = self.clf.coef_.nbytes + self.clf.intercept_.nbytes
        return int(res_bytes + readout_bytes)
