"""Small non-spiking reservoir baseline (Leaky Echo State Network) with matched connectivity."""

from __future__ import annotations

import time
import numpy as np

from src.spiking.lif_reservoir import encode_sensory_inputs
from src.spiking.readout import BalancedRidgeReadout


class LeakyESNClassifier:
    """Non-spiking continuous reservoir baseline matched to the LIF reservoir size and graph."""

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int = 256,
        tau_ms: float = 80.0,
        dt_ms: float = 20.0,
        p_rec: float = 0.15,
        spectral_radius: float = 0.90,
        input_scale: float = 0.65,
        n_temporal_bins: int = 4,
        readout_alpha: float = 10.0,
        seed: int = 0,
    ) -> None:
        self.n_inputs = n_inputs
        self.n_encoded_inputs = n_inputs * 3
        self.n_neurons = n_neurons
        self.tau_ms = float(tau_ms)
        self.dt_ms = float(dt_ms)
        self.p_rec = float(p_rec)
        self.spectral_radius = float(spectral_radius)
        self.input_scale = float(input_scale)
        self.n_temporal_bins = int(n_temporal_bins)
        self.readout_alpha = float(readout_alpha)
        self.seed = int(seed)

        conn_rng = np.random.default_rng(self.seed * 10007 + 42)
        self.W_in = conn_rng.normal(
            0.0,
            self.input_scale / np.sqrt(self.n_encoded_inputs),
            size=(self.n_encoded_inputs, self.n_neurons),
        ).astype(np.float32)

        mask_rec = conn_rng.random((self.n_neurons, self.n_neurons)) < self.p_rec
        np.fill_diagonal(mask_rec, False)
        W_rec = conn_rng.normal(0.0, 1.0, size=(self.n_neurons, self.n_neurons)).astype(np.float32) * mask_rec
        eigvals = np.linalg.eigvals(W_rec)
        max_eig = float(np.max(np.abs(eigvals))) + 1e-8
        self.W_rec = (W_rec * (self.spectral_radius / max_eig)).astype(np.float32)

        self.alpha = float(np.exp(-self.dt_ms / self.tau_ms))
        self.recurrent_fanout = mask_rec.sum(axis=1).astype(np.float32)
        self.macs_per_window = float(
            128 * (self.n_encoded_inputs * self.n_neurons + mask_rec.sum()) + self.n_neurons * 6
        )

        self.clf = BalancedRidgeReadout(alpha=self.readout_alpha)
        self.train_time_sec: float = 0.0

    def _run_states(self, X: np.ndarray) -> np.ndarray:
        B, T, _ = X.shape
        N = self.n_neurons
        X_enc = encode_sensory_inputs(X)
        I_ff = np.einsum("btc,cn->btn", X_enc, self.W_in, optimize=True)
        h = np.zeros((B, N), dtype=np.float32)
        states = np.zeros((B, T, N), dtype=np.float32)
        a = self.alpha
        one_minus_a = 1.0 - a

        for t in range(T):
            pre = I_ff[:, t, :] + (h @ self.W_rec)
            h = a * h + one_minus_a * np.tanh(pre)
            states[:, t, :] = h

        bin_edges = np.linspace(0, T, self.n_temporal_bins + 1, dtype=int)
        feats = []
        for b_idx in range(self.n_temporal_bins):
            s, e = bin_edges[b_idx], bin_edges[b_idx + 1]
            feats.append(states[:, s:e, :].mean(axis=1))
        feats.append(states.mean(axis=1))
        feats.append(states.std(axis=1))
        return np.concatenate(feats, axis=1).astype(np.float32)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LeakyESNClassifier":
        t0 = time.perf_counter()
        F_train = self._run_states(X_train)
        self.clf.fit(F_train, y_train)
        self.train_time_sec = time.perf_counter() - t0
        return self

    def predict_and_measure(self, X_eval: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        F_eval = self._run_states(X_eval)
        preds = self.clf.predict(F_eval)
        eval_sec = time.perf_counter() - t0
        return preds, eval_sec

    @property
    def model_size_bytes(self) -> int:
        assert self.clf.coef_ is not None and self.clf.intercept_ is not None
        return int(
            self.W_in.nbytes
            + self.W_rec.nbytes
            + self.clf.coef_.nbytes
            + self.clf.intercept_.nbytes
        )
