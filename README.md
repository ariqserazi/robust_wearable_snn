# Robust Wearable Activity Recognition with Multi-Timescale Spiking Networks

Code for our Brain-Inspired Computing project testing whether heterogeneous membrane time constants in a spiking reservoir improve robustness to missing sensor readings on the UCI HAR dataset.

## Project Structure

- `src/data/` - UCI HAR inertial signal loader, participant-separated train/val/test split logic, and dropout mask generation
- `src/baselines/` - Window feature classifier and non-spiking leaky ESN baseline
- `src/spiking/` - Leaky integrate-and-fire (LIF) reservoir (homogeneous vs. heterogeneous time constants) and linear ridge readout
- `src/evaluation/` - Accuracy/Macro-F1 metrics and matplotlib plotting functions
- `scripts/` - Scripts to run data prep, runtime check, validation tuning, main experiments, and plots
- `results/` - Saved JSON experiment logs and generated plots

## Setup and Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline steps in order:

```bash
python3 scripts/prepare_data.py
python3 scripts/pilot_runtime.py
python3 scripts/tune_hyperparams.py
python3 scripts/run_experiments.py
python3 scripts/make_plots.py
```

The first script downloads the official UCI HAR dataset archive into `data/raw/` and writes the processed splits and corruption masks to `data/processed/`. Output metrics and figures are saved under `results/`.

## Summary of Results (5 Seeds)

| Missing Data Condition | Heterogeneous LIF | Homogeneous LIF | Non-Spiking ESN | Feature Classifier |
| :--- | :---: | :---: | :---: | :---: |
| 0% (Clean) | 93.01% | 93.92% | 92.46% | 88.19% |
| 10% Gap | 92.37% | 93.16% | 92.01% | 80.39% |
| 20% Gap | 90.46% | 90.11% | 88.28% | 70.08% |
| 30% Gap | 85.96% | 83.36% | 75.08% | 56.92% |
| 40% Gap | 78.51% | 74.65% | 59.61% | 45.26% |
| Single-Axis Channel Loss | 85.27% | 85.96% | 86.05% | 69.83% |

## References

1. UCI HAR Dataset: https://archive.ics.uci.edu/dataset/240/humanactivityrecognitionusingsmartphones
2. Li et al. (2023), *Efficient human activity recognition with spatio-temporal spiking neural networks*, Frontiers in Neuroscience.
3. Perez-Nieves et al. (2021), *Neural heterogeneity promotes robust learning*, Nature Communications.
4. Agrawal et al. (2026), *A Spiking Heterogeneous Harmonic Resonate-and-Fire State Space Model for Time Series*, PMLR.
