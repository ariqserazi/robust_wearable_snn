#!/usr/bin/env python3
"""Milestone 4: Generate publication-quality plots, mechanistic raster demo, and markdown summary."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import build_har_splits
from src.evaluation.plots import (
    plot_activity_cost_tradeoff,
    plot_channel_loss,
    plot_intermittent_robustness,
    plot_neural_dynamics_demo,
    plot_paired_seed_and_per_class,
)
from src.spiking.lif_reservoir import SpikingLIFReservoir, encode_sensory_inputs


def main() -> None:
    with open(ROOT / "configs" / "default_experiment.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results_path = ROOT / "results" / "experiment_results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    f1 = plot_intermittent_robustness(results, fig_dir)
    f2 = plot_paired_seed_and_per_class(results, fig_dir)
    f3 = plot_channel_loss(results, fig_dir)
    f4 = plot_activity_cost_tradeoff(results, fig_dir)

    # Build single-window mechanistic demo data (Walking window with a 30% middle gap)
    splits = build_har_splits(
        raw_dir=ROOT / cfg["data"]["raw_dir"],
        include_total_acc=bool(cfg["data"]["include_total_acc"]),
        val_participant_count=int(cfg["data"]["val_participant_count"]),
    )
    walking_indices = np.where(splits.y_test == 0)[0]
    w_idx = int(walking_indices[0])
    x_clean = splits.X_test[w_idx].copy()  # (128, C)
    x_corrupt = x_clean.copy()
    gap_start, gap_end = 45, 83  # 38 steps = 30% of 128
    x_corrupt[gap_start:gap_end, :] = 0.0

    het_tau = results["selected_hyperparameters"]["heterogeneous_tau_triplet_ms"]
    res_cfg = cfg["reservoir"]
    res_het = SpikingLIFReservoir(
        n_inputs=len(splits.channel_names),
        n_neurons=int(res_cfg["n_neurons"]),
        tau_mode="heterogeneous",
        tau_values_ms=het_tau,
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

    # Run single window and capture traces
    B_in = encode_sensory_inputs(x_corrupt[None, :, :])
    T = 128
    N = res_het.n_neurons
    I_ff = np.einsum("btc,cn->btn", B_in, res_het.W_in) + res_het.bias
    v = np.zeros((1, N), dtype=np.float32)
    r = np.zeros((1, N), dtype=np.float32)
    spikes = np.zeros((T, N), dtype=np.float32)
    traces = np.zeros((T, N), dtype=np.float32)
    alpha = res_het.alpha[None, :]
    one_minus_alpha = 1.0 - alpha

    for t in range(T):
        I_rec = r @ res_het.W_rec
        drive = I_ff[:, t, :] + I_rec
        v = alpha * v + one_minus_alpha * (drive * 2.2)
        fired = (v >= res_het.v_th).astype(np.float32)
        spikes[t, :] = fired[0]
        v = np.maximum(0.0, v - fired * res_het.v_th)
        r = alpha * r + one_minus_alpha * fired
        traces[t, :] = r[0]

    demo_data = {
        "x_corrupt": x_corrupt,
        "gap_bounds": (gap_start, gap_end),
        "spikes_het": spikes,
        "traces_het": traces,
        "tau_m": res_het.tau_m,
        "tau_triplet_ms": het_tau,
        "activity_label": "Walking",
        "channel_indices": {ch: i for i, ch in enumerate(splits.channel_names)},
    }
    f5 = plot_neural_dynamics_demo(demo_data, fig_dir)

    print("=== Generated All 5 Figures ===")
    for p in [f1, f2, f3, f4, f5]:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
