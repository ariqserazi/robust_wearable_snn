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


def main():
    with open(ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    splits = build_har_splits(
        raw_dir=ROOT / cfg["data"]["raw_dir"],
        include_total_acc=cfg["data"]["include_total_acc"],
        val_participant_count=cfg["data"]["val_participant_count"],
    )
    res_cfg = cfg["reservoir"]
    pilot_records = []

    print("Running runtime and firing-rate check...")
    for n_neurons in res_cfg["pilot_sizes"]:
        for tau_mode, tau_val in [
            ("homogeneous", 80.0),
            ("heterogeneous", [25.0, 100.0, 400.0]),
        ]:
            reservoir = SpikingLIFReservoir(
                n_inputs=len(splits.channel_names),
                n_neurons=n_neurons,
                tau_mode=tau_mode,
                tau_values_ms=tau_val,
                dt_ms=cfg["data"]["dt_ms"],
                p_rec=res_cfg["p_rec"],
                excitatory_fraction=res_cfg["excitatory_fraction"],
                spectral_radius=res_cfg["spectral_radius"],
                input_scale=res_cfg["input_scale"],
                v_th=res_cfg["v_th"],
                bias_current=res_cfg["bias_current"],
                n_temporal_bins=res_cfg["n_temporal_bins"],
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
                "val_accuracy": round(val_metrics["accuracy"], 4),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
                "model_size_kb": round(clf.model_size_bytes / 1024.0, 2),
            }
            pilot_records.append(rec)
            print(
                f"  N={n_neurons} ({tau_mode}): time={rec['fit_time_sec']:.2f}s, "
                f"rate={rec['mean_firing_rate_hz']:.2f} Hz, val_f1={rec['val_macro_f1']:.4f}"
            )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runtime_pilot.json", "w") as f:
        json.dump(pilot_records, f, indent=2)


if __name__ == "__main__":
    main()
