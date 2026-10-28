#!/usr/bin/env python3
"""Milestone 3: Paired-seed evaluation across clean, intermittent gaps, and channel loss (Slides 8-10)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.esn_baseline import LeakyESNClassifier
from src.baselines.feature_baseline import WindowFeatureClassifier
from src.data.corruption import (
    apply_corruption_mask,
    generate_channel_loss_masks,
    generate_intermittent_gap_masks,
)
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
    seeds = [int(s) for s in cfg["seeds"]]
    fracs = [float(f) for f in cfg["corruption"]["intermittent_fractions"]]

    # Load validation-selected time constants if available, else use sensible defaults
    tuning_path = ROOT / "results" / "tuning_results.json"
    if tuning_path.exists():
        with open(tuning_path, "r", encoding="utf-8") as f:
            tuned = json.load(f)
        hom_tau = float(tuned["selected_homogeneous"]["tau_ms"])
        hom_alpha = float(tuned["selected_homogeneous"]["readout_alpha"])
        het_tau = [float(x) for x in tuned["selected_heterogeneous"]["tau_triplet_ms"]]
        het_alpha = float(tuned["selected_heterogeneous"]["readout_alpha"])
    else:
        hom_tau, hom_alpha = 80.0, 10.0
        het_tau, het_alpha = [25.0, 100.0, 400.0], 10.0

    # Generate shared test corruption masks
    n_test, window_len, n_channels = splits.X_test.shape
    intermittent_masks = generate_intermittent_gap_masks(
        n_windows=n_test,
        window_length=window_len,
        n_channels=n_channels,
        missing_fractions=fracs,
        seed=int(cfg["corruption"]["mask_seed"]),
    )
    channel_masks = generate_channel_loss_masks(
        n_windows=n_test,
        window_length=window_len,
        channel_names=splits.channel_names,
    )

    # Pre-build masked test sets once so every model sees identical corrupted inputs
    X_test_intermittent = {
        f"{f:.2f}": apply_corruption_mask(splits.X_test, intermittent_masks[round(f, 2)])
        for f in fracs
    }
    X_test_channel = {
        ch: apply_corruption_mask(splits.X_test, channel_masks[ch])
        for ch in splits.channel_names
    }

    results: dict[str, object] = {
        "seeds": seeds,
        "intermittent_fractions": [f"{f:.2f}" for f in fracs],
        "channel_names": splits.channel_names,
        "activity_names": splits.activity_names,
        "selected_hyperparameters": {
            "homogeneous_tau_ms": hom_tau,
            "homogeneous_readout_alpha": hom_alpha,
            "heterogeneous_tau_triplet_ms": het_tau,
            "heterogeneous_readout_alpha": het_alpha,
        },
        "models": {
            "Heterogeneous LIF": {"seeds": []},
            "Homogeneous LIF": {"seeds": []},
            "Non-Spiking ESN": {"seeds": []},
            "Feature Classifier": {"seeds": []},
        },
    }

    print("=== Milestone 3: Running Paired-Seed Experiments (Train Clean -> Test Clean & Corrupted) ===")
    for seed in seeds:
        print(f"\n--- Paired Seed {seed} ---")
        # 1. Heterogeneous LIF (Experimental)
        res_het = SpikingLIFReservoir(
            n_inputs=n_channels,
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
            seed=seed,
        )
        clf_het = SpikingReservoirClassifier(res_het, readout_alpha=het_alpha).fit(
            splits.X_train_full, splits.y_train_full
        )

        # 2. Homogeneous LIF (Control)
        res_hom = SpikingLIFReservoir(
            n_inputs=n_channels,
            n_neurons=int(res_cfg["n_neurons"]),
            tau_mode="homogeneous",
            tau_values_ms=hom_tau,
            dt_ms=float(cfg["data"]["dt_ms"]),
            p_rec=float(res_cfg["p_rec"]),
            excitatory_fraction=float(res_cfg["excitatory_fraction"]),
            spectral_radius=float(res_cfg["spectral_radius"]),
            input_scale=float(res_cfg["input_scale"]),
            v_th=float(res_cfg["v_th"]),
            bias_current=float(res_cfg["bias_current"]),
            n_temporal_bins=int(res_cfg["n_temporal_bins"]),
            seed=seed,
        )
        clf_hom = SpikingReservoirClassifier(res_hom, readout_alpha=hom_alpha).fit(
            splits.X_train_full, splits.y_train_full
        )

        # 3. Non-Spiking ESN Baseline
        clf_esn = LeakyESNClassifier(
            n_inputs=n_channels,
            n_neurons=int(res_cfg["n_neurons"]),
            tau_ms=hom_tau,
            dt_ms=float(cfg["data"]["dt_ms"]),
            p_rec=float(res_cfg["p_rec"]),
            spectral_radius=float(res_cfg["spectral_radius"]),
            input_scale=float(res_cfg["input_scale"]),
            n_temporal_bins=int(res_cfg["n_temporal_bins"]),
            readout_alpha=hom_alpha,
            seed=seed,
        ).fit(splits.X_train_full, splits.y_train_full)

        # 4. Feature Classifier Baseline
        clf_feat = WindowFeatureClassifier(alpha=10.0, seed=seed).fit(
            splits.X_train_full, splits.y_train_full
        )

        # Evaluate Spiking Models
        for model_name, clf_spk in [
            ("Heterogeneous LIF", clf_het),
            ("Homogeneous LIF", clf_hom),
        ]:
            seed_entry = {
                "seed": seed,
                "train_time_sec": clf_spk.train_time_sec,
                "model_size_bytes": clf_spk.model_size_bytes,
                "intermittent": {},
                "channel_loss": {},
            }
            for f_key, X_eval in X_test_intermittent.items():
                preds, r_out, eval_sec = clf_spk.predict_and_measure(X_eval)
                m = compute_classification_metrics(splits.y_test, preds, splits.activity_names)
                m.update(
                    {
                        "eval_time_sec": eval_sec,
                        "mean_spikes_per_window": r_out.mean_spikes_per_window,
                        "mean_firing_rate_hz": r_out.mean_firing_rate_hz,
                        "mean_synops_per_window": r_out.mean_synops_per_window,
                    }
                )
                seed_entry["intermittent"][f_key] = m

            for ch_name, X_eval in X_test_channel.items():
                preds, r_out, eval_sec = clf_spk.predict_and_measure(X_eval)
                m = compute_classification_metrics(splits.y_test, preds, splits.activity_names)
                m.update(
                    {
                        "eval_time_sec": eval_sec,
                        "mean_spikes_per_window": r_out.mean_spikes_per_window,
                        "mean_synops_per_window": r_out.mean_synops_per_window,
                    }
                )
                seed_entry["channel_loss"][ch_name] = m

            results["models"][model_name]["seeds"].append(seed_entry)
            print(
                f"  {model_name:18s} | Clean F1={seed_entry['intermittent']['0.00']['macro_f1']*100:5.2f}% "
                f"| 20% Gap F1={seed_entry['intermittent']['0.20']['macro_f1']*100:5.2f}% "
                f"| 30% Gap F1={seed_entry['intermittent']['0.30']['macro_f1']*100:5.2f}% "
                f"| Spikes/win={seed_entry['intermittent']['0.00']['mean_spikes_per_window']:6.1f}"
            )

        # Evaluate Conventional Baselines
        for model_name, clf_base in [
            ("Non-Spiking ESN", clf_esn),
            ("Feature Classifier", clf_feat),
        ]:
            seed_entry = {
                "seed": seed,
                "train_time_sec": clf_base.train_time_sec,
                "model_size_bytes": clf_base.model_size_bytes,
                "intermittent": {},
                "channel_loss": {},
            }
            for f_key, X_eval in X_test_intermittent.items():
                preds, eval_sec = clf_base.predict_and_measure(X_eval)
                m = compute_classification_metrics(splits.y_test, preds, splits.activity_names)
                m["eval_time_sec"] = eval_sec
                seed_entry["intermittent"][f_key] = m

            for ch_name, X_eval in X_test_channel.items():
                preds, eval_sec = clf_base.predict_and_measure(X_eval)
                m = compute_classification_metrics(splits.y_test, preds, splits.activity_names)
                m["eval_time_sec"] = eval_sec
                seed_entry["channel_loss"][ch_name] = m

            results["models"][model_name]["seeds"].append(seed_entry)
            print(
                f"  {model_name:18s} | Clean F1={seed_entry['intermittent']['0.00']['macro_f1']*100:5.2f}% "
                f"| 20% Gap F1={seed_entry['intermittent']['0.20']['macro_f1']*100:5.2f}% "
                f"| 30% Gap F1={seed_entry['intermittent']['0.30']['macro_f1']*100:5.2f}%"
            )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete experiment results to {out_dir / 'experiment_results.json'}")


if __name__ == "__main__":
    main()
