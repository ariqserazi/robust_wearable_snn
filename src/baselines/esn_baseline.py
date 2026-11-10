import time
import numpy as np

from src.spiking.lif_reservoir import encode_sensory_inputs
from src.spiking.readout import BalancedRidgeReadout


class LeakyESNClassifier:
    def __init__(
        self,
        n_inputs,
        n_neurons=128,
        tau_ms=80.0,
        dt_ms=20.0,
        p_rec=0.15,
        spectral_radius=0.90,
        input_scale=0.65,
        n_temporal_bins=4,
        readout_alpha=10.0,
        seed=0,
    ):
        self.n_inputs = n_inputs
        self.n_encoded_inputs = n_inputs * 3
        self.n_neurons = int(n_neurons)
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
        W_rec = (
            conn_rng.normal(0.0, 1.0, size=(self.n_neurons, self.n_neurons)).astype(np.float32)
            * mask_rec
        )
        eigvals = np.linalg.eigvals(W_rec)
        max_eig = float(np.max(np.abs(eigvals))) + 1e-8
        self.W_rec = (W_rec * (self.spectral_radius / max_eig)).astype(np.float32)

        self.alpha = float(np.exp(-self.dt_ms / self.tau_ms))
        self.recurrent_fanout = mask_rec.sum(axis=1).astype(np.float32)
        self.macs_per_window = float(
            128 * (self.n_encoded_inputs * self.n_neurons + mask_rec.sum()) + self.n_neurons * 6
        )

        self.clf = BalancedRidgeReadout(alpha=self.readout_alpha)
        self.train_time_sec = 0.0

    def _run_states(self, X):
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

    def fit(self, X_train, y_train):
        t0 = time.perf_counter()
        F_train = self._run_states(X_train)
        self.clf.fit(F_train, y_train)
        self.train_time_sec = time.perf_counter() - t0
        return self

    def predict_and_measure(self, X_eval):
        t0 = time.perf_counter()
        F_eval = self._run_states(X_eval)
        preds = self.clf.predict(F_eval)
        eval_sec = time.perf_counter() - t0
        return preds, eval_sec

    @property
    def model_size_bytes(self):
        return int(
            self.W_in.nbytes
            + self.W_rec.nbytes
            + self.clf.coef_.nbytes
            + self.clf.intercept_.nbytes
        )
