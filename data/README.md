# Dataset & Channel Simulation Documentation

## Overview
This directory stores synthetic wireless channel realizations, topology coordinates, and optimization supervisory labels used for training and evaluating the Graph Neural Network (GNN).

Because wireless propagation environments depend on physical terrain, antenna geometry, and user mobility, reproducible evaluation relies on standardized statistical channel models defined by 3GPP and ITU-R.

## Channel Generation Model
Topologies are generated on-the-fly or cached via `DatasetGenerator` in `src/training.py`.

### 1. Spatial Coordinates
- **Geometry**: Single Base Station (BS) centered at $(0, 0)$.
- **Cell Coverage**: Annular disc with inner radius $R_{\min} = 20\text{ m}$ (near-field exclusion zone) and cell radius $R_{\max} = 500\text{ m}$.
- **Spatial Distribution**: Area-uniform random distribution:
  $$r_i = \sqrt{u_i \cdot (R_{\max}^2 - R_{\min}^2) + R_{\min}^2}, \quad u_i \sim \mathcal{U}(0, 1)$$
  $$\theta_i \sim \mathcal{U}(0, 2\pi)$$
  $$x_i = r_i \cos(\theta_i), \quad y_i = r_i \sin(\theta_i)$$

### 2. Large-Scale Path Loss
Log-distance path loss model:
$$PL(d_i) [\text{dB}] = PL_0 + 10 \alpha \log_{10}\left(\frac{d_i}{d_0}\right)$$
- Reference distance $d_0 = 1.0\text{ m}$
- Reference path loss $PL_0 = 38.4\text{ dB}$ (at $f_c = 2.4\text{ GHz}$)
- Path-loss exponent $\alpha = 3.5$ (urban microcell propagation)

### 3. Small-Scale Fading
Independent and identically distributed (i.i.d.) complex Gaussian Rayleigh fading:
$$h_{\text{fading}, i} \sim \mathcal{CN}(0, \sigma_f^2) = \frac{1}{\sqrt{2}}(X + jY), \quad X, Y \sim \mathcal{N}(0, 1)$$
Composite channel coefficient:
$$h_i = \sqrt{10^{-PL(d_i)/10}} \cdot h_{\text{fading}, i}$$
Channel power gain:
$$g_i = |h_i|^2$$

### 4. Thermal Noise & Power
- Thermal noise spectral density $N_0 = -174\text{ dBm/Hz}$
- Noise figure $F = 5.0\text{ dB}$
- Total bandwidth $B = 10\text{ MHz}$ divided across $S=5$ slots ($B_s = 2\text{ MHz}$ per slot)
- Total BS transmit power $P = 1.0\text{ W}$ (30 dBm)

## Supervisory Labels
Training labels are generated without feature leakage by solving the combinatorial sum-rate utility problem:
- Pair adjacency matrix $\mathbf{Y} \in \{0, 1\}^{N \times N}$
- Slot assignment vector $\mathbf{s} \in \{0, \dots, S-1\}^N$

Labels are derived using:
- **Exhaustive Partition Enumeration** for $N \le 10$ users (guaranteed global optimum).
- **Edmonds' Blossom Maximum-Weight Matching** for $N > 10$ users.

## Reproducibility
All generation steps accept an explicit integer random seed:
- Training Split: `seed = 100` to `299`
- Validation Split: `seed = 500` to `539`
- Test Benchmark Split: `seed = 2000` to `2049`

## Optional CSV Input

Synthetic simulation remains the default. Optional CSV analysis can be run with:

```text
python main.py --mode compare --data-source csv --dataset data/example_users.csv --users 4
```

The CSV is a **user-level scenario table**: one row represents one user in one complete network scenario. Required columns are:

| Column | Type | Unit / meaning |
|---|---|---|
| `scenario_id` | integer/string | Scenario identifier used for leakage-safe splitting |
| `user_id` | integer | User identifier, unique inside each scenario |
| `x`, `y` | numeric | User coordinates in meters |
| `distance` | numeric | Distance from the base station in meters |
| `channel_gain` | numeric | Non-negative linear power gain $|h|^2$, not dB |
| `min_rate_req_bps` | numeric | Minimum required rate in bits per second |

The loader checks file existence, required columns, numeric values, positive distances, non-negative linear gains, duplicate user IDs, and the configured number of users per scenario. Missing values are rejected with a clear error rather than silently imputed at the input boundary.

The current CSV path derives the continuous rate target from the same configured bandwidth, transmit power, and noise model used by the simulator. It does not invent real-world labels. The target is suitable for the classical user-rate regression comparison; a CSV without labels is not treated as a measured supervised ground truth.

Feature preprocessing is fitted inside each classical model pipeline on the training split only and then reused for the held-out test split and inference. Scenario IDs are used only for splitting and are never model features. The CSV loader does not silently replace the GNN synthetic training path: `--mode train` remains the original synthetic GNN workflow, while `--mode compare` and `--mode clustering` support both sources.

An example CSV should be clearly marked as synthetic or user-provided; no measurement claims are made by this project.

`example_users.csv` is included only as a small synthetic-format demonstration. It is not real measurement data.
