#!/usr/bin/env python3
"""Milestone 2: Equal-budget hyperparameter tuning on clean training-participant validation split (Slides 4, 8, 9)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.corruption import apply_corruption_mask, generate_intermittent_gap_masks
from src.data.dataset import build_har_splits
from src.evaluation.metrics import compute_classification_metrics
from src.spiking.lif_reservoir import SpikingLIFReservoir
from src.spiking.readout import BalancedRidgeReadout


def evaluate_reservoir_across_alphas(
    reservoir: SpikingLIFReservoir,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val_clean: np.ndarray,
    X_val_gap20: np.ndarray,
    X_val_gap30: np.ndarray,
    y_val: np.ndarray,
    activity_names: list[str],
    alpha_grid: list[float],
) -> list[dict[str, float]]:
    """Run reservoir once on fit and val splits, then evaluate all readout regularization values."""
    F_fit = reservoir.run_numpy(X_fit).features
    F_val_clean = reservoir.run_numpy(X_val_clean).features
    F_val_gap20 = reservoir.run_numpy(X_val_gap20).features
    F_val_gap30 = reservoir.run_numpy(X_val_gap30).features

    out_records = []
    for alpha in alpha_grid:
        readout = BalancedRidgeReadout(alpha=float(alpha)).fit(F_fit, y_fit)
        p_clean = readout.predict(F_val_clean)
        p_gap20 = readout.predict(F_val_gap20)
        p_gap30 = readout.predict(F_val_gap30)

        f1_clean = float(compute_classification_metrics(y_val, p_clean, activity_names)["macro_f1"])
        f1_gap20 = float(compute_classification_metrics(y_val, p_gap20, activity_names)["macro_f1"])
        f1_gap30 = float(compute_classification_metrics(y_val, p_gap30, activity_names)["macro_f1"])

        out_records.append(
            {
                "readout_alpha": float(alpha),
                "val_f1_clean": round(f1_clean, 4),
                "val_f1_gap20": round(f1_gap20, 4),
                "val_f1_gap30": round(f1_gap30, 4),
            }
        )
    return out_records


def main() -> None:
    with open(ROOT / "configs" / "default_experiment.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits = build_har_splits(
        raw_dir=ROOT / cfg["data"]["raw_dir"],
        include_total_acc=bool(cfg["data"]["include_total_acc"]),
        val_participant_count=int(cfg["data"]["val_participant_count"]),
    )
    res_cfg = cfg["reservoir"]
    tune_cfg = cfg["tuning"]

    # Create validation corruption masks for diagnostic logging, while selecting strictly on clean validation data (Slide 9)
    val_masks = generate_intermittent_gap_masks(
        n_windows=len(splits.X_val),
        window_length=splits.X_val.shape[1],
        n_channels=splits.X_val.shape[2],
        missing_fractions=[0.0, 0.20, 0.30],
        seed=999,
    )
    X_val_clean = splits.X_val
    X_val_gap20 = apply_corruption_mask(splits.X_val, val_masks[0.20])
    X_val_gap30 = apply_corruption_mask(splits.X_val, val_masks[0.30])

    print("=== Milestone 2: Equal-Budget Clean Validation Tuning (24 Configs Each, Training Participants Only) ===")

    tuning_summary: dict[str, object] = {"homogeneous": [], "heterogeneous": []}

    # 1. Tune Homogeneous LIF (6 tau candidates x 4 alpha candidates = 24 configs, selected on clean val F1)
    best_hom_score = -1.0
    best_hom_cfg = None
    for tau_ms in tune_cfg["homogeneous_tau_candidates_ms"]:
        reservoir = SpikingLIFReservoir(
            n_inputs=len(splits.channel_names),
            n_neurons=int(res_cfg["n_neurons"]),
            tau_mode="homogeneous",
            tau_values_ms=float(tau_ms),
            dt_ms=float(cfg["data"]["dt_ms"]),
            p_rec=float(res_cfg["p_rec"]),
            excitatory_fraction=float(res_cfg["excitatory_fraction"]),
            spectral_radius=float(res_cfg["spectral_radius"]),
            input_scale=float(res_cfg["input_scale"]),
            v_th=float(res_cfg["v_th"]),
            bias_current=float(res_cfg["bias_current"]),
            n_temporal_bins=int(res_cfg["n_temporal_bins"]),
            seed=0,
        )
        recs = evaluate_reservoir_across_alphas(
            reservoir,
            splits.X_fit,
            splits.y_fit,
            X_val_clean,
            X_val_gap20,
            X_val_gap30,
            splits.y_val,
            splits.activity_names,
            tune_cfg["readout_alpha_grid"],
        )
        for r in recs:
            entry = {"tau_ms": float(tau_ms), **r}
            tuning_summary["homogeneous"].append(entry)
            if entry["val_f1_clean"] > best_hom_score:
                best_hom_score = entry["val_f1_clean"]
                best_hom_cfg = entry

    # 2. Tune Heterogeneous LIF (6 tau triplets x 4 alpha candidates = 24 configs, selected on clean val F1)
    best_het_score = -1.0
    best_het_cfg = None
    for tau_triplet in tune_cfg["heterogeneous_tau_candidates_ms"]:
        reservoir = SpikingLIFReservoir(
            n_inputs=len(splits.channel_names),
            n_neurons=int(res_cfg["n_neurons"]),
            tau_mode="heterogeneous",
            tau_values_ms=list(tau_triplet),
            dt_ms=float(cfg["data"]["dt_ms"]),
            p_rec=float(res_cfg["p_rec"]),
            excitatory_fraction=float(res_cfg["excitatory_fraction"]),
            spectral_radius=float(res_cfg["spectral_radius"]),
            input_scale=float(res_cfg["input_scale"]),
            v_th=float(res_cfg["v_th"]),
            bias_current=float(res_cfg["bias_current"]),
            n_temporal_bins=int(res_cfg["n_temporal_bins"]),
            seed=0,
        )
        recs = evaluate_reservoir_across_alphas(
            reservoir,
            splits.X_fit,
            splits.y_fit,
            X_val_clean,
            X_val_gap20,
            X_val_gap30,
            splits.y_val,
            splits.activity_names,
            tune_cfg["readout_alpha_grid"],
        )
        for r in recs:
            entry = {"tau_triplet_ms": [float(x) for x in tau_triplet], **r}
            tuning_summary["heterogeneous"].append(entry)
            if entry["val_f1_clean"] > best_het_score:
                best_het_score = entry["val_f1_clean"]
                best_het_cfg = entry

    tuning_summary["selected_homogeneous"] = best_hom_cfg
    tuning_summary["selected_heterogeneous"] = best_het_cfg

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(tuning_summary, f, indent=2)

    print(f"Selected Homogeneous Config (by Clean Val F1):   {best_hom_cfg}")
    print(f"Selected Heterogeneous Config (by Clean Val F1): {best_het_cfg}")
    print(f"Saved validation tuning results to {out_dir / 'tuning_results.json'}")


if __name__ == "__main__":
    main()
