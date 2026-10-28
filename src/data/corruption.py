"""Generate and apply deterministic missing-data corruption masks (Slide 9)."""

from __future__ import annotations

import numpy as np


def generate_intermittent_gap_masks(
    n_windows: int,
    window_length: int,
    n_channels: int,
    missing_fractions: list[float],
    seed: int = 202610,
) -> dict[float, np.ndarray]:
    """Generate boolean masks (True = present, False = missing/zeroed) for contiguous block dropout.

    For each window `w` and missing fraction `p`, removes a single contiguous block
    of length `round(window_length * p)` across all sensor channels.
    """
    masks: dict[float, np.ndarray] = {}
    for frac in missing_fractions:
        frac_key = round(float(frac), 2)
        mask = np.ones((n_windows, window_length, n_channels), dtype=bool)
        gap_len = int(round(window_length * frac_key))
        if gap_len > 0:
            # Deterministic RNG per fraction so masks are 100% reproducible across models
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
    channel_names: list[str],
) -> dict[str, np.ndarray]:
    """Generate boolean masks where one sensor axis is removed for the full window."""
    n_channels = len(channel_names)
    masks: dict[str, np.ndarray] = {}
    for ch_idx, ch_name in enumerate(channel_names):
        mask = np.ones((n_windows, window_length, n_channels), dtype=bool)
        mask[:, :, ch_idx] = False
        masks[ch_name] = mask
    return masks


def apply_corruption_mask(X_standardized: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace missing standardized entries with 0.0 under the shared protocol (Slide 9)."""
    return np.where(mask, X_standardized, 0.0).astype(np.float32)
