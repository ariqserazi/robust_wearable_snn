#!/usr/bin/env python3
"""Milestone 1: Early runtime and firing-rate pilot (Slides 7, 11, 13)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import build_har_splits
from src.evaluation.metrics import compute_classification_metrics
from src.spiking.lif_reservoir import SpikingLIFReservoir
from src.spiking.readout import SpikingReservoirClassifier


def main() -> None:
    with open(ROOT / "configs" / "default_experiment.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits = build_har_splits(
        raw_dir=ROOT / cfg["data"]["raw_dir"],
        include_total_acc=bool(cfg["data"]["include_total_acc"]),
        val_participant_count=int(cfg["data"]["val_participant_count"]),
    )
    res_cfg = cfg["reservoir"]
    pilot_records = []

    print("=== Milestone 1: Runtime & Firing-Rate Pilot ===")
    for n_neurons in res_cfg["pilot_sizes"]:
        for tau_mode, tau_val in [
            ("homogeneous", 80.0),
            ("heterogeneous", [25.0, 100.0, 400.0]),
        ]:
            reservoir = SpikingLIFReservoir(
                n_inputs=len(splits.channel_names),
                n_neurons=int(n_neurons),
                tau_mode=tau_mode,
                tau_values_ms=tau_val,
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
            clf = SpikingReservoirClassifier(reservoir, readout_alpha=10.0)
            clf.fit(splits.X_fit, splits.y_fit)
            preds_val, val_out, eval_sec = clf.predict_and_measure(splits.X_val)
            val_metrics = compute_classification_metrics(
                splits.y_val, preds_val, splits.activity_names
            )

            rec = {
                "n_neurons": int(n_neurons),
                "tau_mode": tau_mode,
                "tau_values_ms": tau_val,
                "fit_time_sec": round(clf.train_time_sec, 4),
                "val_eval_time_sec": round(eval_sec, 4),
                "ms_per_window": round((eval_sec / len(splits.X_val)) * 1000.0, 4),
                "mean_firing_rate_hz": round(val_out.mean_firing_rate_hz, 2),
                "mean_spikes_per_window": round(val_out.mean_spikes_per_window, 1),
                "mean_synops_per_window": round(val_out.mean_synops_per_window, 1),
                "silent_neuron_fraction": round(val_out.silent_neuron_fraction, 4),
                "hyperactive_fraction": round(val_out.hyperactive_fraction, 4),
                "val_accuracy": round(float(val_metrics["accuracy"]), 4),
                "val_macro_f1": round(float(val_metrics["macro_f1"]), 4),
                "model_size_kb": round(clf.model_size_bytes / 1024.0, 2),
            }
            pilot_records.append(rec)
            print(
                f"N={n_neurons:3d} | {tau_mode:13s} | Fit={rec['fit_time_sec']:.2f}s | "
                f"Rate={rec['mean_firing_rate_hz']:5.2f} Hz | Silent={rec['silent_neuron_fraction']*100:4.1f}% | "
                f"Val Macro-F1={rec['val_macro_f1']*100:5.2f}%"
            )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runtime_pilot.json", "w", encoding="utf-8") as f:
        json.dump(pilot_records, f, indent=2)
    print(f"Saved pilot report to {out_dir / 'runtime_pilot.json'}")


if __name__ == "__main__":
    main()
