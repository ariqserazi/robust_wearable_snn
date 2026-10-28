"""Download and extract the official UCI HAR Dataset."""

from __future__ import annotations

import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

UCI_HAR_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
]


def _get_ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl._create_unverified_context()


def ensure_uci_har_downloaded(raw_dir: str | Path = "data/raw") -> Path:
    """Download and extract the UCI HAR dataset if not already present.

    Returns the path to the extracted `UCI HAR Dataset` directory.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    extracted_dir = raw_path / "UCI HAR Dataset"

    if (extracted_dir / "train" / "Inertial Signals").exists() and (
        extracted_dir / "test" / "Inertial Signals"
    ).exists():
        return extracted_dir

    zip_path = raw_path / "UCI_HAR_Dataset.zip"
    if not zip_path.exists():
        ctx = _get_ssl_context()
        last_err = None
        for url in UCI_HAR_URLS:
            try:
                print(f"Downloading UCI HAR dataset from {url} ...")
                with urllib.request.urlopen(url, context=ctx, timeout=60) as resp, open(
                    zip_path, "wb"
                ) as out_f:
                    shutil.copyfileobj(resp, out_f)
                break
            except Exception as exc:  # pragma: no cover
                last_err = exc
        else:
            raise RuntimeError(f"Failed to download UCI HAR dataset: {last_err}")

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_path)

    # Handle nested zip if downloaded from the new UCI static mirror
    nested_zip = raw_path / "UCI HAR Dataset.zip"
    if not extracted_dir.exists() and nested_zip.exists():
        with zipfile.ZipFile(nested_zip, "r") as zf:
            zf.extractall(raw_path)

    if not extracted_dir.exists():
        raise FileNotFoundError(f"Expected directory not found at {extracted_dir}")

    return extracted_dir
