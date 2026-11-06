import numpy as np


def generate_intermittent_gap_masks(
    n_windows: int,
    window_length: int,
    n_channels: int,
    missing_fractions: list,
    seed: int = 202610,
):
    masks = {}
    for frac in missing_fractions:
        frac_key = round(float(frac), 2)
        mask = np.ones((n_windows, window_length, n_channels), dtype=bool)
        gap_len = int(round(window_length * frac_key))
        if gap_len > 0:
            rng = np.random.default_rng(seed + int(frac_key * 1000))
            max_start = window_length - gap_len
            starts = rng.integers(0, max_start + 1, size=n_windows)
            for w in range(n_windows):
                s = starts[w]
                mask[w, s : s + gap_len, :] = False
        masks[frac_key] = mask
    return masks


def generate_channel_loss_masks(
    n_windows: int,
    window_length: int,
    channel_names: list,
):
    n_channels = len(channel_names)
    masks = {}
    for ch_idx, ch_name in enumerate(channel_names):
        mask = np.ones((n_windows, window_length, n_channels), dtype=bool)
        mask[:, :, ch_idx] = False
        masks[ch_name] = mask
    return masks


def apply_corruption_mask(X_standardized: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, X_standardized, 0.0).astype(np.float32)
