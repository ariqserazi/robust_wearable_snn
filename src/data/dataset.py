from pathlib import Path
import numpy as np

from src.data.download_uci_har import ensure_uci_har_downloaded

ACTIVITY_NAMES = [
    "Walking",
    "Upstairs",
    "Downstairs",
    "Sitting",
    "Standing",
    "Lying down",
]

CORE_6_CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

ALL_9_CHANNELS = CORE_6_CHANNELS + [
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


class HARDataSplits:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _load_split_raw(dataset_dir, split, channels):
    signals_dir = Path(dataset_dir) / split / "Inertial Signals"
    channel_arrays = []
    for ch in channels:
        fpath = signals_dir / f"{ch}_{split}.txt"
        arr = np.loadtxt(fpath, dtype=np.float32)
        channel_arrays.append(arr)

    X = np.stack(channel_arrays, axis=-1)
    y = np.loadtxt(Path(dataset_dir) / split / f"y_{split}.txt", dtype=np.int64) - 1
    subj = np.loadtxt(Path(dataset_dir) / split / f"subject_{split}.txt", dtype=np.int64)
    return X, y, subj


def build_har_splits(
    raw_dir="data/raw",
    include_total_acc=True,
    val_participant_count=5,
    split_seed=42,
):
    dataset_dir = ensure_uci_har_downloaded(raw_dir)
    channels = ALL_9_CHANNELS if include_total_acc else CORE_6_CHANNELS

    X_train_raw, y_train_full, subj_train_full = _load_split_raw(dataset_dir, "train", channels)
    X_test_raw, y_test, subj_test = _load_split_raw(dataset_dir, "test", channels)

    train_subjects = np.unique(subj_train_full)
    test_subjects = np.unique(subj_test)
    assert len(np.intersect1d(train_subjects, test_subjects)) == 0

    rng = np.random.default_rng(split_seed)
    shuffled_train_subj = rng.permutation(train_subjects)
    val_subjects = np.sort(shuffled_train_subj[:val_participant_count])

    val_mask = np.isin(subj_train_full, val_subjects)
    fit_mask = ~val_mask

    fit_mean = X_train_raw[fit_mask].mean(axis=(0, 1), keepdims=True)
    fit_std = X_train_raw[fit_mask].std(axis=(0, 1), keepdims=True) + 1e-8

    X_fit = (X_train_raw[fit_mask] - fit_mean) / fit_std
    X_val = (X_train_raw[val_mask] - fit_mean) / fit_std

    train_mean = X_train_raw.mean(axis=(0, 1), keepdims=True)
    train_std = X_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train_full = (X_train_raw - train_mean) / train_std
    X_test = (X_test_raw - train_mean) / train_std

    return HARDataSplits(
        X_train_full=X_train_full.astype(np.float32),
        y_train_full=y_train_full,
        subj_train_full=subj_train_full,
        X_fit=X_fit.astype(np.float32),
        y_fit=y_train_full[fit_mask],
        subj_fit=subj_train_full[fit_mask],
        X_val=X_val.astype(np.float32),
        y_val=y_train_full[val_mask],
        subj_val=subj_train_full[val_mask],
        X_test=X_test.astype(np.float32),
        y_test=y_test,
        subj_test=subj_test,
        channel_names=channels,
        activity_names=ACTIVITY_NAMES,
        scaler_mean=train_mean.astype(np.float32),
        scaler_std=train_std.astype(np.float32),
    )
