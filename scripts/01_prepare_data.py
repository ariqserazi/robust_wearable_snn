#!/usr/bin/env python3
"""Milestone 0: Download UCI HAR, build participant-separated splits, and save corruption masks."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.corruption import generate_channel_loss_masks, generate_intermittent_gap_masks
from src.data.dataset import build_har_splits


def main() -> None:
    with open(ROOT / "configs" / "default_experiment.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = ROOT / cfg["data"]["raw_dir"]
    proc_dir = ROOT / cfg["data"]["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    splits = build_har_splits(
        raw_dir=raw_dir,
        include_total_acc=bool(cfg["data"]["include_total_acc"]),
        val_participant_count=int(cfg["data"]["val_participant_count"]),
        split_seed=42,
    )

    n_test, window_len, n_channels = splits.X_test.shape
    intermittent_masks = generate_intermittent_gap_masks(
        n_windows=n_test,
        window_length=window_len,
        n_channels=n_channels,
        missing_fractions=cfg["corruption"]["intermittent_fractions"],
        seed=int(cfg["corruption"]["mask_seed"]),
    )
    channel_masks = generate_channel_loss_masks(
        n_windows=n_test,
        window_length=window_len,
        channel_names=splits.channel_names,
    )

    # Save splits and masks for reproducibility (Slide 13)
    np.savez_compressed(
        proc_dir / "uci_har_splits.npz",
        X_train_full=splits.X_train_full,
        y_train_full=splits.y_train_full,
        subj_train_full=splits.subj_train_full,
        X_fit=splits.X_fit,
        y_fit=splits.y_fit,
        subj_fit=splits.subj_fit,
        X_val=splits.X_val,
        y_val=splits.y_val,
        subj_val=splits.subj_val,
        X_test=splits.X_test,
        y_test=splits.y_test,
        subj_test=splits.subj_test,
        scaler_mean=splits.scaler_mean,
        scaler_std=splits.scaler_std,
    )

    np.savez_compressed(
        proc_dir / "corruption_masks.npz",
        **{f"intermittent_{k:.2f}": v for k, v in intermittent_masks.items()},
        **{f"channel_{k}": v for k, v in channel_masks.items()},
    )

    meta = {
        "n_train_windows": int(splits.X_train_full.shape[0]),
        "n_fit_windows": int(splits.X_fit.shape[0]),
        "n_val_windows": int(splits.X_val.shape[0]),
        "n_test_windows": int(splits.X_test.shape[0]),
        "window_length": int(window_len),
        "n_channels": int(n_channels),
        "channel_names": splits.channel_names,
        "activity_names": splits.activity_names,
        "fit_participants": [int(x) for x in np.unique(splits.subj_fit)],
        "val_participants": [int(x) for x in np.unique(splits.subj_val)],
        "test_participants": [int(x) for x in np.unique(splits.subj_test)],
    }
    with open(proc_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=== UCI HAR Data & Corruption Masks Prepared ===")
    print(f"Train full: {splits.X_train_full.shape} ({len(np.unique(splits.subj_train_full))} participants)")
    print(f"  -> Fit split: {splits.X_fit.shape} ({len(meta['fit_participants'])} participants: {meta['fit_participants']})")
    print(f"  -> Val split: {splits.X_val.shape} ({len(meta['val_participants'])} participants: {meta['val_participants']})")
    print(f"Official Test: {splits.X_test.shape} ({len(meta['test_participants'])} participants: {meta['test_participants']})")
    print(f"Saved processed tensors and masks to: {proc_dir}")


if __name__ == "__main__":
    main()
