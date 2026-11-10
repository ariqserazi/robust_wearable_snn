import time
import numpy as np

from src.spiking.readout import BalancedRidgeReadout


def extract_window_features(X, n_bins=4):
    B, T, C = X.shape
    feats = []

    feats.append(X.mean(axis=1))
    feats.append(X.std(axis=1))
    feats.append(X.min(axis=1))
    feats.append(X.max(axis=1))
    feats.append(np.sqrt(np.mean(np.square(X), axis=1)))
    feats.append(np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1))

    diffs = np.diff(X, axis=1)
    feats.append(np.mean(np.abs(diffs), axis=1))
    feats.append(np.std(diffs, axis=1))
    signs = np.sign(X)
    zcr = np.mean(signs[:, 1:, :] != signs[:, :-1, :], axis=1)
    feats.append(zcr)

    bin_edges = np.linspace(0, T, n_bins + 1, dtype=int)
    for b_idx in range(n_bins):
        s, e = bin_edges[b_idx], bin_edges[b_idx + 1]
        feats.append(X[:, s:e, :].mean(axis=1))
        feats.append(X[:, s:e, :].std(axis=1))

    return np.concatenate(feats, axis=1).astype(np.float32)


class WindowFeatureClassifier:
    def __init__(self, alpha=10.0, seed=0):
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.clf = BalancedRidgeReadout(alpha=self.alpha)
        self.train_time_sec = 0.0

    def fit(self, X_train, y_train):
        t0 = time.perf_counter()
        F_train = extract_window_features(X_train)
        self.clf.fit(F_train, y_train)
        self.train_time_sec = time.perf_counter() - t0
        return self

    def predict_and_measure(self, X_eval):
        t0 = time.perf_counter()
        F_eval = extract_window_features(X_eval)
        preds = self.clf.predict(F_eval)
        eval_sec = time.perf_counter() - t0
        return preds, eval_sec

    @property
    def model_size_bytes(self):
        return int(self.clf.coef_.nbytes + self.clf.intercept_.nbytes)
