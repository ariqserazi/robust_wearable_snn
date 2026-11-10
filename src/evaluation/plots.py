from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "Heterogeneous LIF": "#1B5E20",
    "Homogeneous LIF": "#D95F02",
    "Non-Spiking ESN": "#2B7BBA",
    "Feature Classifier": "#756BB1",
}

MARKERS = {
    "Heterogeneous LIF": "o",
    "Homogeneous LIF": "o",
    "Non-Spiking ESN": "s",
    "Feature Classifier": "^",
}

LINESTYLES = {
    "Heterogeneous LIF": "-",
    "Homogeneous LIF": "-",
    "Non-Spiking ESN": "--",
    "Feature Classifier": "--",
}


def setup_plot_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_intermittent_robustness(results, out_dir):
    setup_plot_style()
    fracs = [float(f) for f in results["intermittent_fractions"]]
    pcts = [int(round(f * 100)) for f in fracs]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    for model_name, m_data in results["models"].items():
        seeds_list = m_data["seeds"]
        f1_mat = np.array(
            [[s["intermittent"][f"{f:.2f}"]["macro_f1"] * 100.0 for f in fracs] for s in seeds_list]
        )
        mean_f1 = f1_mat.mean(axis=0)
        std_f1 = f1_mat.std(axis=0)

        ax.plot(
            pcts,
            mean_f1,
            label=model_name,
            color=COLORS[model_name],
            marker=MARKERS[model_name],
            linestyle=LINESTYLES[model_name],
            linewidth=2.2,
        )
        ax.fill_between(
            pcts, mean_f1 - std_f1, mean_f1 + std_f1, color=COLORS[model_name], alpha=0.15
        )

    ax.set_title("Macro-F1 vs Missing Sensor Data")
    ax.set_xlabel("Missing Readings per Window (%)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xticks(pcts)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)

    ax2 = axes[1]
    for model_name, m_data in results["models"].items():
        seeds_list = m_data["seeds"]
        acc_mat = np.array(
            [[s["intermittent"][f"{f:.2f}"]["accuracy"] * 100.0 for f in fracs] for s in seeds_list]
        )
        acc_drop = acc_mat - acc_mat[:, [0]]
        mean_drop = acc_drop.mean(axis=0)
        std_drop = acc_drop.std(axis=0)

        ax2.plot(
            pcts,
            mean_drop,
            label=model_name,
            color=COLORS[model_name],
            marker=MARKERS[model_name],
            linestyle=LINESTYLES[model_name],
            linewidth=2.2,
        )
        ax2.fill_between(
            pcts, mean_drop - std_drop, mean_drop + std_drop, color=COLORS[model_name], alpha=0.15
        )

    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    ax2.set_title("Accuracy Drop vs Clean Score")
    ax2.set_xlabel("Missing Readings per Window (%)")
    ax2.set_ylabel("Accuracy Change (%)")
    ax2.set_xticks(pcts)
    ax2.grid(True, linestyle=":", alpha=0.5)

    out_path = Path(out_dir) / "fig1_intermittent_robustness.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_paired_seed_and_per_class(results, out_dir):
    setup_plot_style()
    fracs = [float(f) for f in results["intermittent_fractions"]]
    pcts = [f"{int(round(f * 100))}%" for f in fracs]

    het_seeds = results["models"]["Heterogeneous LIF"]["seeds"]
    hom_seeds = results["models"]["Homogeneous LIF"]["seeds"]

    diff_mat = np.array(
        [
            [
                (het["intermittent"][f"{f:.2f}"]["macro_f1"] - hom["intermittent"][f"{f:.2f}"]["macro_f1"])
                * 100.0
                for f in fracs
            ]
            for het, hom in zip(het_seeds, hom_seeds)
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))

    ax = axes[0]
    x = np.arange(len(fracs))
    mean_diff = diff_mat.mean(axis=0)
    std_diff = diff_mat.std(axis=0)

    ax.bar(
        x,
        mean_diff,
        yerr=std_diff,
        capsize=4,
        color="#1B5E20",
        alpha=0.75,
        width=0.55,
    )
    for seed_idx in range(diff_mat.shape[0]):
        jitter = (seed_idx - 2) * 0.06
        ax.scatter(
            x + jitter,
            diff_mat[seed_idx],
            color="#111111",
            s=28,
            zorder=4,
            alpha=0.75,
        )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(pcts)
    ax.set_title("Paired Difference (Heterogeneous - Homogeneous)")
    ax.set_xlabel("Missing Data Fraction")
    ax.set_ylabel("Macro-F1 Difference (%)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

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

    ax2.barh(y_pos + height / 2, het_class_f1, height, label="Heterogeneous LIF", color=COLORS["Heterogeneous LIF"], alpha=0.9)
    ax2.barh(y_pos - height / 2, hom_class_f1, height, label="Homogeneous LIF", color=COLORS["Homogeneous LIF"], alpha=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(activities)
    ax2.invert_yaxis()
    ax2.set_xlabel("F1 Score at 30% Gap (%)")
    ax2.set_title("Per-Class F1 Under 30% Dropout")
    ax2.set_xlim(0, 105)
    ax2.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    out_path = Path(out_dir) / "fig2_paired_seed_differences.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_channel_loss(results, out_dir):
    setup_plot_style()
    channels = results["channel_names"]
    models = ["Heterogeneous LIF", "Homogeneous LIF", "Non-Spiking ESN", "Feature Classifier"]

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    x = np.arange(len(channels) + 1)
    width = 0.20

    for idx, m in enumerate(models):
        seed_list = results["models"][m]["seeds"]
        ch_scores = []
        for ch in channels:
            vals = [s["channel_loss"][ch]["macro_f1"] * 100.0 for s in seed_list]
            ch_scores.append(np.mean(vals))
        ch_scores.append(float(np.mean(ch_scores)))

        offset = (idx - 1.5) * width
        ax.bar(x + offset, ch_scores, width, label=m, color=COLORS[m], alpha=0.88)

    pretty_channel_map = {
        "body_acc_x": r"Body Acc$_x$",
        "body_acc_y": r"Body Acc$_y$",
        "body_acc_z": r"Body Acc$_z$",
        "body_gyro_x": r"Body Gyro$_x$",
        "body_gyro_y": r"Body Gyro$_y$",
        "body_gyro_z": r"Body Gyro$_z$",
        "total_acc_x": r"Total Acc$_x$",
        "total_acc_y": r"Total Acc$_y$",
        "total_acc_z": r"Total Acc$_z$",
    }
    labels = [pretty_channel_map.get(ch, ch) for ch in channels] + ["Mean"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(30, 100)
    ax.set_title("Single-Axis Channel Loss Evaluation", pad=28)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=True)

    out_path = Path(out_dir) / "fig3_channel_loss.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_activity_cost_tradeoff(results, out_dir):
    setup_plot_style()
    fracs = [float(f) for f in results["intermittent_fractions"]]
    pcts = [int(round(f * 100)) for f in fracs]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    for m in ["Heterogeneous LIF", "Homogeneous LIF"]:
        spk_mat = np.array(
            [
                [s["intermittent"][f"{f:.2f}"]["mean_spikes_per_window"] for f in fracs]
                for s in results["models"][m]["seeds"]
            ]
        )
        mean_spk = spk_mat.mean(axis=0)
        std_spk = spk_mat.std(axis=0)
        ax.plot(pcts, mean_spk, marker="o", linewidth=2.2, color=COLORS[m], label=m)
        ax.fill_between(pcts, mean_spk - std_spk, mean_spk + std_spk, color=COLORS[m], alpha=0.15)

    ax.set_title("Spike Count per Window")
    ax.set_xlabel("Missing Readings (%)")
    ax.set_ylabel("Mean Spikes / Window")
    ax.set_xticks(pcts)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True)

    ax2 = axes[1]
    for m in ["Heterogeneous LIF", "Homogeneous LIF"]:
        seeds = results["models"][m]["seeds"]
        synops = [s["intermittent"]["0.30"]["mean_synops_per_window"] / 1000.0 for s in seeds]
        f1s = [s["intermittent"]["0.30"]["macro_f1"] * 100.0 for s in seeds]
        ax2.scatter(synops, f1s, color=COLORS[m], s=65, alpha=0.85, label=m)

    ax2.set_title("Synaptic Operations vs Macro-F1 (30% Gap)")
    ax2.set_xlabel("Estimated SynOps per Window (thousands)")
    ax2.set_ylabel("Macro-F1 at 30% Gap (%)")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(frameon=True, loc="lower right")

    out_path = Path(out_dir) / "fig4_activity_cost_tradeoff.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_neural_dynamics_demo(demo_data, out_dir):
    setup_plot_style()
    time_sec = np.arange(128) * 0.02
    x_corrupt = demo_data["x_corrupt"]
    gap_start, gap_end = demo_data["gap_bounds"]
    spikes_het = demo_data["spikes_het"]
    traces_het = demo_data["traces_het"]
    tau_m = demo_data["tau_m"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))

    ax0 = axes[0, 0]
    t_decay = np.linspace(0, 1.0, 200)
    tau_fast, tau_med, tau_slow = demo_data["tau_triplet_ms"]
    ax0.plot(t_decay * 1000, np.exp(-t_decay * 1000 / tau_fast), color="#1B9E77", lw=2.0, label=rf"Fast ($\tau_m = {tau_fast:.0f}$ ms)")
    ax0.plot(t_decay * 1000, np.exp(-t_decay * 1000 / tau_med), color="#D95F02", lw=2.0, label=rf"Medium ($\tau_m = {tau_med:.0f}$ ms)")
    ax0.plot(t_decay * 1000, np.exp(-t_decay * 1000 / tau_slow), color="#7570B3", lw=2.0, label=rf"Slow ($\tau_m = {tau_slow:.0f}$ ms)")
    ax0.set_title("Exponential Decay Curves")
    ax0.set_xlabel("Time (ms)")
    ax0.set_ylabel(r"Retained Signal $\exp(-t / \tau_m)$")
    ax0.grid(True, linestyle=":", alpha=0.5)
    ax0.legend(frameon=True)

    ax1 = axes[0, 1]
    pretty_ch = {
        "body_acc_x": r"Body Acc$_x$",
        "body_acc_y": r"Body Acc$_y$",
        "body_gyro_z": r"Body Gyro$_z$",
    }
    for ch_label in ["body_acc_x", "body_acc_y", "body_gyro_z"]:
        idx = demo_data["channel_indices"][ch_label]
        ax1.plot(time_sec, x_corrupt[:, idx], lw=1.5, label=pretty_ch[ch_label])
    ax1.axvspan(gap_start * 0.02, gap_end * 0.02, color="#FFCDD2", alpha=0.5, label="30% Missing Gap")
    ax1.set_title(f"Sensor Input ({demo_data['activity_label']}) with 30% Gap")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Standardized Reading")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True)

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
    ax2.scatter(t_idx * 0.02, n_idx, c=colors_raster, s=4.0, alpha=0.85)
    ax2.axvspan(gap_start * 0.02, gap_end * 0.02, color="#FFCDD2", alpha=0.35)
    ax2.axhline(n_third, color="gray", linestyle="--", lw=0.8)
    ax2.axhline(2 * n_third, color="gray", linestyle="--", lw=0.8)
    ax2.set_title("Spike Raster (Sorted by Neuron Timescale)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Neuron Index")
    ax2.set_xlim(0, 2.56)
    ax2.set_ylim(0, N)

    ax3 = axes[1, 1]
    traces_sorted = traces_het[:, order]
    fast_trace = traces_sorted[:, :n_third].mean(axis=1)
    med_trace = traces_sorted[:, n_third : 2 * n_third].mean(axis=1)
    slow_trace = traces_sorted[:, 2 * n_third :].mean(axis=1)

    ax3.plot(time_sec, fast_trace / (fast_trace.max() + 1e-8), color="#1B9E77", lw=2.0, label="Fast Neurons")
    ax3.plot(time_sec, med_trace / (med_trace.max() + 1e-8), color="#D95F02", lw=2.0, label="Medium Neurons")
    ax3.plot(time_sec, slow_trace / (slow_trace.max() + 1e-8), color="#7570B3", lw=2.2, label="Slow Neurons")
    ax3.axvspan(gap_start * 0.02, gap_end * 0.02, color="#FFCDD2", alpha=0.45, label="Missing Gap")
    ax3.set_title("Normalized Population Traces Across Gap")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Normalized Activity")
    ax3.set_xlim(0, 2.56)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="lower left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    out_path = Path(out_dir) / "fig5_neural_dynamics_demo.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
