"""Generate publication-quality figures for the Brain-Inspired Computing project (Slide 12-13)."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Consistent academic color palette matching the presentation theme
COLORS = {
    "Heterogeneous LIF": "#1B5E20",   # Deep forest green (Experimental)
    "Homogeneous LIF": "#D95F02",     # Warm orange (Control)
    "Non-Spiking ESN": "#377EB8",     # Steel blue (Baseline 2)
    "Feature Classifier": "#7570B3",  # Muted purple (Baseline 1)
}


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def plot_intermittent_robustness(results: dict, out_dir: Path) -> Path:
    """Figure 1: Macro-F1 and Accuracy Retention under Intermittent Block Gaps (0%-40%)."""
    setup_plot_style()
    fracs = [float(f) for f in results["intermittent_fractions"]]
    pcts = [int(round(f * 100)) for f in fracs]
    models = ["Heterogeneous LIF", "Homogeneous LIF", "Non-Spiking ESN", "Feature Classifier"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Panel A: Absolute Macro-F1 vs Missing %
    ax = axes[0]
    for m in models:
        f1_matrix = np.array(
            [[ seed_res["intermittent"][f"{f:.2f}"]["macro_f1"] for f in fracs ]
             for seed_res in results["models"][m]["seeds"]]
        ) * 100.0
        mean_f1 = f1_matrix.mean(axis=0)
        std_f1 = f1_matrix.std(axis=0)
        marker = "o" if "LIF" in m else ("s" if "ESN" in m else "^")
        lw = 2.6 if "LIF" in m else 1.8
        ls = "-" if "LIF" in m else "--"
        ax.plot(pcts, mean_f1, label=m, color=COLORS[m], marker=marker, linewidth=lw, linestyle=ls)
        ax.fill_between(pcts, mean_f1 - std_f1, mean_f1 + std_f1, color=COLORS[m], alpha=0.15)

    ax.set_title("A. Recognition Quality Under Intermittent Gaps")
    ax.set_xlabel("Missing Sensor Readings per Window (%)")
    ax.set_ylabel("Macro-F1 Score (%)")
    ax.set_xticks(pcts)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, loc="lower left")

    # Panel B: Relative Accuracy Loss from each model's Clean (0%) Score
    ax2 = axes[1]
    for m in models:
        acc_matrix = np.array(
            [[ seed_res["intermittent"][f"{f:.2f}"]["accuracy"] for f in fracs ]
             for seed_res in results["models"][m]["seeds"]]
        ) * 100.0
        loss_matrix = acc_matrix - acc_matrix[:, :1]  # relative to 0% column
        mean_loss = loss_matrix.mean(axis=0)
        std_loss = loss_matrix.std(axis=0)
        marker = "o" if "LIF" in m else ("s" if "ESN" in m else "^")
        lw = 2.6 if "LIF" in m else 1.8
        ls = "-" if "LIF" in m else "--"
        ax2.plot(pcts, mean_loss, label=m, color=COLORS[m], marker=marker, linewidth=lw, linestyle=ls)
        ax2.fill_between(pcts, mean_loss - std_loss, mean_loss + std_loss, color=COLORS[m], alpha=0.15)

    ax2.axhline(0.0, color="gray", linewidth=1, linestyle=":")
    ax2.set_title("B. Accuracy Loss Relative to Clean Baseline")
    ax2.set_xlabel("Missing Sensor Readings per Window (%)")
    ax2.set_ylabel("Accuracy Change vs. 0% Clean (percentage points)")
    ax2.set_xticks(pcts)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Robustness to Intermittent Sensor Block Dropout (5 Paired Random Seeds)", fontsize=14, fontweight="bold", y=1.03)
    out_path = out_dir / "fig1_intermittent_robustness.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_paired_seed_and_per_class(results: dict, out_dir: Path) -> Path:
    """Figure 2: Paired differences (Heterogeneous - Homogeneous) across seeds & Per-Class F1 at 30% gap."""
    setup_plot_style()
    fracs = [float(f) for f in results["intermittent_fractions"]]
    pcts = [f"{int(round(f * 100))}%" for f in fracs]

    het_seeds = results["models"]["Heterogeneous LIF"]["seeds"]
    hom_seeds = results["models"]["Homogeneous LIF"]["seeds"]
    n_seeds = len(het_seeds)

    diff_f1 = np.zeros((n_seeds, len(fracs)))
    for s_idx in range(n_seeds):
        for f_idx, f in enumerate(fracs):
            k = f"{f:.2f}"
            diff_f1[s_idx, f_idx] = (
                het_seeds[s_idx]["intermittent"][k]["macro_f1"]
                - hom_seeds[s_idx]["intermittent"][k]["macro_f1"]
            ) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    # Panel A: Paired Delta Macro-F1 across dropout levels with individual seed dots
    ax = axes[0]
    x = np.arange(len(fracs))
    means = diff_f1.mean(axis=0)
    stds = diff_f1.std(axis=0)
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#2E7D32", alpha=0.80, width=0.55, label="Mean ± SD (Paired)")
    for s_idx in range(n_seeds):
        jitter = (s_idx - (n_seeds - 1) / 2.0) * 0.06
        ax.scatter(x + jitter, diff_f1[s_idx], color="#111111", s=28, zorder=5, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(pcts)
    ax.set_title("A. Paired Gain: Heterogeneous minus Homogeneous LIF")
    ax.set_xlabel("Intermittent Missing Data Fraction")
    ax.set_ylabel("Δ Macro-F1 (percentage points)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    for bar, m in zip(bars, means):
        if m >= 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                m + 0.6,
                f"{m:+.2f}%",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                1.2,
                f"{m:+.2f}%",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    # Panel B: Per-Class F1 at 30% Intermittent Gap
    ax2 = axes[1]
    activities = results["activity_names"]
    y_pos = np.arange(len(activities))
    height = 0.36

    het_class_f1 = np.array(
        [[s["intermittent"]["0.30"]["per_class_f1"][act] for act in activities] for s in het_seeds]
    ).mean(axis=0) * 100.0
    hom_class_f1 = np.array(
        [[s["intermittent"]["0.30"]["per_class_f1"][act] for act in activities] for s in hom_seeds]
    ).mean(axis=0) * 100.0

    ax2.barh(y_pos + height / 2, het_class_f1, height, label="Heterogeneous LIF (Mixed τ)", color=COLORS["Heterogeneous LIF"], alpha=0.9)
    ax2.barh(y_pos - height / 2, hom_class_f1, height, label="Homogeneous LIF (Single τ)", color=COLORS["Homogeneous LIF"], alpha=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(activities)
    ax2.invert_yaxis()
    ax2.set_xlabel("Class F1 Score at 30% Missing Gap (%)")
    ax2.set_title("B. Per-Activity Recognition Under 30% Dropout")
    ax2.set_xlim(0, 105)
    ax2.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    fig.suptitle("Matched Paired-Seed Analysis & Activity-Specific Breakdown (Slide 8 & 10)", fontsize=14, fontweight="bold", y=1.03)
    out_path = out_dir / "fig2_paired_seed_differences.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_channel_loss(results: dict, out_dir: Path) -> Path:
    """Figure 3: Single-Axis Sensor Channel Loss Evaluation (Slide 9)."""
    setup_plot_style()
    channels = results["channel_names"]
    models = ["Heterogeneous LIF", "Homogeneous LIF", "Non-Spiking ESN", "Feature Classifier"]

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    x = np.arange(len(channels) + 1)
    width = 0.20

    for idx, m in enumerate(models):
        seed_list = results["models"][m]["seeds"]
        ch_scores = []
        for ch in channels:
            vals = [s["channel_loss"][ch]["macro_f1"] * 100.0 for s in seed_list]
            ch_scores.append(np.mean(vals))
        ch_scores.append(float(np.mean(ch_scores)))  # Summary mean across axes

        offset = (idx - 1.5) * width
        ax.bar(x + offset, ch_scores, width, label=m, color=COLORS[m], alpha=0.88)

    labels = channels + ["Mean Across\nAll Axes"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Macro-F1 Score (%)")
    ax.set_ylim(30, 100)
    ax.set_title("Recognition Robustness Under Full-Window Single-Axis Channel Loss (Slide 9)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=True)

    out_path = out_dir / "fig3_channel_loss.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_activity_cost_tradeoff(results: dict, out_dir: Path) -> Path:
    """Figure 4: Computational Activity Cost (Spikes & SynOps per Window) vs. Accuracy (Slide 10)."""
    setup_plot_style()
    fracs = [float(f) for f in results["intermittent_fractions"]]
    pcts = [int(round(f * 100)) for f in fracs]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    # Panel A: Mean Spike Count per Window across Missing Data %
    ax = axes[0]
    for m in ["Heterogeneous LIF", "Homogeneous LIF"]:
        spk_mat = np.array(
            [[s["intermittent"][f"{f:.2f}"]["mean_spikes_per_window"] for f in fracs]
             for s in results["models"][m]["seeds"]]
        )
        mean_spk = spk_mat.mean(axis=0)
        std_spk = spk_mat.std(axis=0)
        ax.plot(pcts, mean_spk, marker="o", linewidth=2.5, color=COLORS[m], label=m)
        ax.fill_between(pcts, mean_spk - std_spk, mean_spk + std_spk, color=COLORS[m], alpha=0.15)

    ax.set_title("A. Total Spike Count per 2.56s Window")
    ax.set_xlabel("Missing Sensor Readings (%)")
    ax.set_ylabel("Mean Spikes / Window (128 Neurons)")
    ax.set_xticks(pcts)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True)

    # Panel B: Estimated Synaptic Operations (SynOps) vs Macro-F1 at 30% Gap
    ax2 = axes[1]
    for m in ["Heterogeneous LIF", "Homogeneous LIF"]:
        seeds = results["models"][m]["seeds"]
        synops = [s["intermittent"]["0.30"]["mean_synops_per_window"] / 1000.0 for s in seeds]
        f1s = [s["intermittent"]["0.30"]["macro_f1"] * 100.0 for s in seeds]
        ax2.scatter(synops, f1s, color=COLORS[m], s=80, alpha=0.85, label=f"{m} (Seeds 0–4)")
        ax2.scatter(
            [np.mean(synops)],
            [np.mean(f1s)],
            color=COLORS[m],
            s=220,
            marker="*",
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
        )

    ax2.set_title("B. Activity Cost vs. Recognition at 30% Dropout")
    ax2.set_xlabel("Estimated Synaptic Operations per Window (×10³ SynOps)")
    ax2.set_ylabel("Macro-F1 at 30% Missing Data (%)")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(frameon=True, loc="lower right")

    fig.suptitle("Computational Activity Cost & Synaptic Operations Analysis (Slide 10)", fontsize=14, fontweight="bold", y=1.03)
    out_path = out_dir / "fig4_activity_cost_tradeoff.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_neural_dynamics_demo(demo_data: dict, out_dir: Path) -> Path:
    """Figure 5: Mechanistic 4-panel visualization of Fast/Medium/Slow spiking dynamics bridging a gap."""
    setup_plot_style()
    time_sec = np.arange(128) * 0.02  # 50 Hz -> 2.56 seconds
    x_corrupt = demo_data["x_corrupt"]          # (128, C)
    gap_start, gap_end = demo_data["gap_bounds"]
    spikes_het = demo_data["spikes_het"]        # (128, N)
    traces_het = demo_data["traces_het"]        # (128, N)
    tau_m = demo_data["tau_m"]                  # (N,)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))

    # Panel A: Theoretical Exponential Decay Curves of Selected Timescales (Slide 5)
    ax0 = axes[0, 0]
    t_decay = np.linspace(0, 1.0, 200)
    tau_fast, tau_med, tau_slow = demo_data["tau_triplet_ms"]
    ax0.plot(t_decay * 1000, np.exp(-t_decay * 1000 / tau_fast), color="#1B9E77", lw=2.5, label=f"Fast (τ ≈ {tau_fast:.0f} ms)")
    ax0.plot(t_decay * 1000, np.exp(-t_decay * 1000 / tau_med), color="#D95F02", lw=2.5, label=f"Medium (τ ≈ {tau_med:.0f} ms)")
    ax0.plot(t_decay * 1000, np.exp(-t_decay * 1000 / tau_slow), color="#7570B3", lw=2.5, label=f"Slow (τ ≈ {tau_slow:.0f} ms)")
    ax0.set_title("A. Multi-Timescale Memory Decay Kernels exp(-t / τ)")
    ax0.set_xlabel("Time Elapsed After Input Impulse (ms)")
    ax0.set_ylabel("Retained Subthreshold / Trace Signal")
    ax0.grid(True, linestyle=":", alpha=0.5)
    ax0.legend(frameon=True)

    # Panel B: Corrupted Sensor Input Window (Walking with 30% Missing Gap)
    ax1 = axes[0, 1]
    for ch_i, ch_label in enumerate(["body_acc_x", "body_acc_y", "body_gyro_z"]):
        idx = demo_data["channel_indices"][ch_label]
        ax1.plot(time_sec, x_corrupt[:, idx], lw=1.6, label=ch_label)
    ax1.axvspan(gap_start * 0.02, gap_end * 0.02, color="#FFCDD2", alpha=0.55, label="30% Missing Sensor Gap")
    ax1.set_title(f"B. Wearable Sensor Stream ({demo_data['activity_label']}) with 30% Dropout Gap")
    ax1.set_xlabel("Window Time (seconds)")
    ax1.set_ylabel("Standardized Sensor Reading")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9, frameon=True)

    # Panel C: Spike Raster Ordered by Neuron Memory Timescale (Fast -> Medium -> Slow)
    ax2 = axes[1, 0]
    order = np.argsort(tau_m)
    spikes_sorted = spikes_het[:, order]
    N = len(tau_m)
    n_third = N // 3

    t_idx, n_idx = np.where(spikes_sorted > 0)
    colors_raster = np.where(
        n_idx < n_third,
        "#1B9E77",
        np.where(n_idx < 2 * n_third, "#D95F02", "#7570B3"),
    )
    ax2.scatter(t_idx * 0.02, n_idx, c=colors_raster, s=4.5, alpha=0.85)
    ax2.axvspan(gap_start * 0.02, gap_end * 0.02, color="#FFCDD2", alpha=0.35)
    ax2.axhline(n_third, color="gray", linestyle="--", lw=0.8)
    ax2.axhline(2 * n_third, color="gray", linestyle="--", lw=0.8)
    ax2.set_title("C. Spike Raster by Timescale (Green=Fast, Orange=Med, Purple=Slow)")
    ax2.set_xlabel("Window Time (seconds)")
    ax2.set_ylabel("Neuron Index (Sorted by Timescale τ)")
    ax2.set_xlim(0, 2.56)
    ax2.set_ylim(0, N)

    # Panel D: Population Memory Traces Bridging the Missing Sensor Gap
    ax3 = axes[1, 1]
    traces_sorted = traces_het[:, order]
    fast_trace = traces_sorted[:, :n_third].mean(axis=1)
    med_trace = traces_sorted[:, n_third : 2 * n_third].mean(axis=1)
    slow_trace = traces_sorted[:, 2 * n_third :].mean(axis=1)

    ax3.plot(time_sec, fast_trace / (fast_trace.max() + 1e-8), color="#1B9E77", lw=2.2, label="Fast Population Trace (Tracks Rapid Steps)")
    ax3.plot(time_sec, med_trace / (med_trace.max() + 1e-8), color="#D95F02", lw=2.2, label="Medium Population Trace")
    ax3.plot(time_sec, slow_trace / (slow_trace.max() + 1e-8), color="#7570B3", lw=2.5, label="Slow Population Trace (Bridges Gap)")
    ax3.axvspan(gap_start * 0.02, gap_end * 0.02, color="#FFCDD2", alpha=0.45, label="Missing Sensor Gap")
    ax3.set_title("D. Normalized Population Memory Traces Across the Gap")
    ax3.set_xlabel("Window Time (seconds)")
    ax3.set_ylabel("Normalized Population Activity")
    ax3.set_xlim(0, 2.56)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="lower left", fontsize=9, frameon=True)

    fig.suptitle("Mechanistic Demonstration: How Multi-Timescale Spiking Memory Bridges Sensor Gaps", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = out_dir / "fig5_neural_dynamics_demo.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
