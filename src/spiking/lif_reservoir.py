"""Leaky Integrate-and-Fire (LIF) recurrent reservoir with homogeneous and heterogeneous timescales.

Supports both fast batched NumPy execution (for full-dataset multi-seed experiments)
and Brian2 simulation (for runtime pilot benchmarking and single-window biophysical inspection).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np


def encode_sensory_inputs(X: np.ndarray) -> np.ndarray:
    """Shared ON/OFF sensory afferent encoding (applied identically to all reservoir models).

    Maps standardized sensor streams x_t in R^C to [x_t, max(0, x_t), max(0, -x_t)] in R^{3C}.
    When sensor readings are missing (x_t = 0), all encoded channels remain strictly 0.0.
    """
    pos = np.maximum(0.0, X)
    neg = np.maximum(0.0, -X)
    return np.concatenate([X, pos, neg], axis=-1).astype(np.float32)


@dataclass
class ReservoirOutput:
    """Outputs and computational cost metrics from running the spiking reservoir."""

    features: np.ndarray            # (N_windows, D_feat) readout feature matrix
    mean_spikes_per_window: float   # Average total spikes per 128-step window
    mean_firing_rate_hz: float      # Average firing rate per neuron in Hz
    mean_synops_per_window: float   # Average synaptic operations per window
    silent_neuron_fraction: float   # Fraction of neurons with 0 spikes across dataset
    hyperactive_fraction: float     # Fraction of neurons spiking > 50% of steps
    runtime_sec: float              # Wall-clock simulation time
    spike_raster: np.ndarray | None = None      # Optional (T, N) or (B, T, N) spike array
    membrane_traces: np.ndarray | None = None   # Optional (T, N) membrane potential array


class SpikingLIFReservoir:
    """Recurrent Leaky Integrate-and-Fire (LIF) Reservoir with paired-seed control."""

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int = 128,
        tau_mode: str = "homogeneous",
        tau_values_ms: float | list[float] = 80.0,
        dt_ms: float = 20.0,
        p_rec: float = 0.15,
        excitatory_fraction: float = 0.80,
        spectral_radius: float = 0.90,
        input_scale: float = 0.65,
        v_th: float = 1.0,
        bias_current: float = 0.15,
        n_temporal_bins: int = 4,
        seed: int = 0,
    ) -> None:
        self.n_inputs = n_inputs
        self.n_encoded_inputs = n_inputs * 3
        self.n_neurons = n_neurons
        self.tau_mode = tau_mode
        self.tau_values_ms = tau_values_ms
        self.dt_ms = float(dt_ms)
        self.p_rec = float(p_rec)
        self.excitatory_fraction = float(excitatory_fraction)
        self.spectral_radius = float(spectral_radius)
        self.input_scale = float(input_scale)
        self.v_th = float(v_th)
        self.bias_current = float(bias_current)
        self.n_temporal_bins = int(n_temporal_bins)
        self.seed = int(seed)

        # 1. Initialize input and recurrent connectivity using a seed-locked RNG
        #    so Homogeneous and Heterogeneous models have IDENTICAL graph & weights.
        conn_rng = np.random.default_rng(self.seed * 10007 + 42)

        # Input projection weights W_in: (n_encoded_inputs, n_neurons)
        self.W_in = conn_rng.normal(
            0.0,
            self.input_scale / np.sqrt(self.n_encoded_inputs),
            size=(self.n_encoded_inputs, self.n_neurons),
        ).astype(np.float32)

        # Excitatory (80%) and Inhibitory (20%) neuron signs (Dale's principle)
        n_exc = int(round(self.n_neurons * self.excitatory_fraction))
        signs = np.ones(self.n_neurons, dtype=np.float32)
        signs[n_exc:] = -1.0
        conn_rng.shuffle(signs)
        self.neuron_signs = signs

        # Sparse recurrent adjacency mask (no self-connections)
        mask_rec = conn_rng.random((self.n_neurons, self.n_neurons)) < self.p_rec
        np.fill_diagonal(mask_rec, False)

        raw_weights = np.abs(conn_rng.normal(0.0, 1.0, size=(self.n_neurons, self.n_neurons))).astype(
            np.float32
        )
        W_rec = raw_weights * mask_rec
        # Apply Dale's law per presynaptic row and balance E/I drive
        W_rec = (self.neuron_signs[:, None] * W_rec).astype(np.float32)

        # Normalize spectral radius
        eigvals = np.linalg.eigvals(W_rec)
        max_eig = float(np.max(np.abs(eigvals))) + 1e-8
        self.W_rec = (W_rec * (self.spectral_radius / max_eig)).astype(np.float32)

        # Outgoing fan-out per neuron for SynOps calculation (recurrent + 6-class readout)
        self.recurrent_fanout = mask_rec.sum(axis=1).astype(np.float32)
        self.total_fanout = self.recurrent_fanout + 6.0

        # Small heterogeneous tonic bias per neuron (fixed across tau_mode for same seed)
        self.bias = conn_rng.uniform(0.04, self.bias_current, size=(1, self.n_neurons)).astype(
            np.float32
        )

        # 2. Configure membrane/synaptic timescales tau_m (the ONLY factor varied!)
        self.tau_m = self._build_tau_vector()
        self.alpha = np.exp(-self.dt_ms / self.tau_m).astype(np.float32)  # (N,)

    def _build_tau_vector(self) -> np.ndarray:
        """Construct the per-neuron time constant vector in ms."""
        if self.tau_mode == "homogeneous":
            if isinstance(self.tau_values_ms, (list, tuple, np.ndarray)):
                tau_val = float(self.tau_values_ms[0])
            else:
                tau_val = float(self.tau_values_ms)
            return np.full(self.n_neurons, tau_val, dtype=np.float32)

        if self.tau_mode == "heterogeneous":
            if not isinstance(self.tau_values_ms, (list, tuple, np.ndarray)) or len(self.tau_values_ms) < 3:
                raise ValueError(
                    "For heterogeneous mode, tau_values_ms must be [tau_fast, tau_med, tau_slow]."
                )
            tau_fast, tau_med, tau_slow = [float(x) for x in self.tau_values_ms[:3]]
            tau_rng = np.random.default_rng(self.seed * 99991 + 7)
            n_third = self.n_neurons // 3
            n_rem = self.n_neurons - 2 * n_third

            # Create 3 distinct subpopulations (fast, medium, slow) with slight log-normal jitter
            fast_pop = tau_fast * np.exp(tau_rng.normal(0.0, 0.10, size=n_third))
            med_pop = tau_med * np.exp(tau_rng.normal(0.0, 0.10, size=n_third))
            slow_pop = tau_slow * np.exp(tau_rng.normal(0.0, 0.10, size=n_rem))

            taus = np.concatenate([fast_pop, med_pop, slow_pop]).astype(np.float32)
            return np.clip(taus, 10.0, 1200.0)

        raise ValueError(f"Unknown tau_mode: {self.tau_mode}")

    def run_numpy(
        self,
        X: np.ndarray,
        return_traces: bool = False,
    ) -> ReservoirOutput:
        """Run batched discrete-time LIF dynamics over windows X of shape (B, T, C)."""
        t0 = time.perf_counter()
        B, T, _ = X.shape
        N = self.n_neurons

        X_enc = encode_sensory_inputs(X)
        # Precompute feedforward input drive for all windows and timesteps: (B, T, N)
        I_ff = np.einsum("btc,cn->btn", X_enc, self.W_in, optimize=True) + self.bias

        v = np.zeros((B, N), dtype=np.float32)
        r = np.zeros((B, N), dtype=np.float32)  # Multi-timescale filtered postsynaptic trace
        spikes = np.zeros((B, T, N), dtype=np.float32)
        traces = np.zeros((B, T, N), dtype=np.float32)
        v_hist = np.zeros((B, T, N), dtype=np.float32) if return_traces else None

        alpha = self.alpha[None, :]              # (1, N)
        one_minus_alpha = 1.0 - alpha            # (1, N)
        # Moderate subthreshold gain normalization so slow neurons integrate multi-step context
        # without requiring higher firing rates than fast neurons
        v_gain = np.power(one_minus_alpha, 0.65).astype(np.float32) * 1.45

        for t in range(T):
            I_rec = r @ self.W_rec
            drive = I_ff[:, t, :] + I_rec
            v = alpha * v + v_gain * drive

            fired = (v >= self.v_th).astype(np.float32)
            spikes[:, t, :] = fired
            # Soft reset preserves subthreshold memory residual
            v = np.maximum(0.0, v - fired * self.v_th)

            # Biophysical postsynaptic trace decays at neuron timescale tau_m
            r = alpha * r + one_minus_alpha * fired + 0.15 * one_minus_alpha * v
            traces[:, t, :] = r

            if return_traces and v_hist is not None:
                v_hist[:, t, :] = v

        # Build readout representation across temporal sub-windows (bins)
        bin_edges = np.linspace(0, T, self.n_temporal_bins + 1, dtype=int)
        bin_traces = []
        for b_idx in range(self.n_temporal_bins):
            t_start, t_end = bin_edges[b_idx], bin_edges[b_idx + 1]
            bin_traces.append(traces[:, t_start:t_end, :].mean(axis=1))

        # Stack bin traces: (B, n_bins, N)
        bin_stack = np.stack(bin_traces, axis=1)
        feat_parts = list(bin_traces)
        # Continuity features: minimum & median across temporal bins (persists when slow memory bridges a gap)
        feat_parts.append(bin_stack.min(axis=1))
        feat_parts.append(np.median(bin_stack, axis=1))
        # Full-window summary statistics
        feat_parts.append(traces.mean(axis=1))
        feat_parts.append(traces.std(axis=1))
        feat_parts.append(spikes.mean(axis=1))
        features = np.concatenate(feat_parts, axis=1).astype(np.float32)

        # Compute neural activity and SynOps metrics
        spikes_per_window_neuron = spikes.sum(axis=1)  # (B, N)
        mean_spikes_per_window = float(spikes_per_window_neuron.sum(axis=1).mean())
        duration_sec = (T * self.dt_ms) / 1000.0
        mean_firing_rate_hz = float(mean_spikes_per_window / (N * duration_sec))

        synops_per_window = spikes_per_window_neuron @ self.total_fanout  # (B,)
        mean_synops_per_window = float(synops_per_window.mean())

        total_spikes_per_neuron = spikes_per_window_neuron.sum(axis=0)  # (N,)
        silent_fraction = float(np.mean(total_spikes_per_neuron == 0))
        hyperactive_fraction = float(np.mean((spikes_per_window_neuron.mean(axis=0) / T) > 0.5))

        runtime_sec = time.perf_counter() - t0
        return ReservoirOutput(
            features=features,
            mean_spikes_per_window=mean_spikes_per_window,
            mean_firing_rate_hz=mean_firing_rate_hz,
            mean_synops_per_window=mean_synops_per_window,
            silent_neuron_fraction=silent_fraction,
            hyperactive_fraction=hyperactive_fraction,
            runtime_sec=runtime_sec,
            spike_raster=spikes if return_traces else None,
            membrane_traces=v_hist if return_traces else None,
        )
