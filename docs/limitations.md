# Assumptions, Limitations & Honest Scope Assessment

## 0. ML Engineering Scope

The project now includes a lightweight educational ML lifecycle: validated synthetic/CSV scenario input, scenario-level splitting, training-only preprocessing, saved classical artifacts, reusable inference, local monitoring, and candidate retraining comparison. The optional FastAPI service exposes rate prediction for a saved Random Forest artifact. It is not a production deployment: authentication, persistence infrastructure, model registry, distributed monitoring, and drift alerting are not implemented.

The classical models use simulation-derived single-user achievable rate as a regression target. They do not establish real-world prediction accuracy. Clustering is exploratory; K-means and DBSCAN groups are not claimed to be optimal communication pairs. The original GNN pairing and PD-NOMA schedule remain the main communication experiment, while SCMA remains a separate detector evaluation.

## 1. Context & Academic Scope
In accordance with rigorous academic integrity, this document explicitly records the physical assumptions, structural abstractions, and technical boundaries of the implemented system. B.Tech / B.E. capstone projects require transparent communication regarding which components are fully simulated, which use established abstractions, and what remains for future research.

---

## 2. Implemented Features & Verification Status

| Component | Status | Description / Scope |
| :--- | :---: | :--- |
| **Cellular Geometry & Path Loss** | **Implemented & Tested** | 2D annular disk ($20\text{ m} \le d \le 500\text{ m}$) with log-distance path loss ($PL_0 = 38.4\text{ dB}$, $\alpha = 3.5$). |
| **Rayleigh Fading Channel** | **Implemented & Tested** | Independent complex Gaussian small-scale fading with channel power gain $g = \|h\|^2$. |
| **PD-NOMA Superposition & SIC** | **Implemented & Tested** | 2-user power-domain NOMA with closed-form SINR and achievable rate computation. |
| **Imperfect SIC Modeling** | **Implemented & Tested** | Parameterized residual interference coefficient $\beta_{\text{SIC}} \in [0, 1)$ modeling imperfect cancellation at the strong user. |
| **SCMA Factor Graph & Codebooks** | **Implemented & Tested** | Canonical regular factor graph matrix $\mathbf{F}_{4 \times 6}$ ($d_v = 2, d_f = 3$) with 4-point geometric complex codebooks (150% overload). |
| **SCMA MPA Multi-User Detector** | **Implemented & Tested** | Iterative belief propagation Message Passing Algorithm (MPA) computing posterior symbol probabilities and empirical SER/BER. |
| **Combinatorial Oracle** | **Implemented & Tested** | Global optimum exhaustive partition search for $N \le 10$; Edmonds' Blossom maximum-weight matching for $N > 10$. |
| **Edge-Aware GNN Architecture** | **Implemented & Tested** | Native PyTorch message passing layers with edge disparity features, pair affinity head, and slot classification head. |
| **Deterministic Post-Processor** | **Implemented & Tested** | Maximum-weight greedy matching on predicted edge affinities, guaranteeing zero duplicates and robust odd-user handling. |
| **Dynamic Slot Assignment** | **Implemented & Tested** | Orthogonal time-frequency slots under strict capacity and BS power constraints. |
| **Automated Test Suite** | **Implemented & Tested** | 27 unit and integration tests passing with 100% success rate under `pytest`. |

---

## 3. Assumptions & Theoretical Boundaries

### 3.1 Single-Cell Downlink Topology
- **Assumption**: A single isolated Base Station is assumed without inter-cell co-channel interference from adjacent cells.
- **Practical Implication**: In dense multi-tier cellular deployments (heterogeneous networks), inter-cell interference from neighboring base stations affects cell-edge users. This can be extended in future work through multi-cell cooperative GNN scheduling.

### 3.2 Quasi-Static Block Fading
- **Assumption**: Wireless channel coefficients $h_i$ remain static over the transmission slot duration ($T_{\text{slot}} \approx 1\text{ ms}$) and vary independently across coherence intervals (block fading model).
- **Practical Implication**: Fast Doppler-induced time-selective fading is not modeled. Users are assumed to be stationary or pedestrian-speed within each transmission block.

### 3.3 SCMA Physical-Layer Scope & Abstraction
- **What is Fully Implemented**:
  - Full regular factor graph $\mathbf{F} \in \{0, 1\}^{4 \times 6}$.
  - Complex multi-dimensional constellation codebook mapping.
  - Multi-user superimposed transmission over subcarriers.
  - Iterative Message Passing Algorithm (MPA) symbol detector with joint probability updates.
  - Empirical Bit Error Rate (BER) and Symbol Error Rate (SER) Monte Carlo simulation.
- **What is Abstracted**:
  - Codebooks are constructed using phase-rotated QPSK unit-energy constellations rather than high-dimensional multidimensional sphere-packing optimized codebooks (e.g., Huawei Star-QAM or genetic-algorithm codebooks).
  - Perfect channel state information (CSI) is assumed at the receiver during MPA iteration.
  - Channel coding (e.g., LDPC or Polar codes) is omitted; raw uncoded transmission is simulated to isolate multi-user detection capabilities.

### 3.4 PD-NOMA Pair Cardinality
- **Assumption**: PD-NOMA groups are restricted to 2 users per resource slot (one weak user and one strong user).
- **Justification**: In practical cellular standards (e.g., 3GPP LTE/5G NR study items on MUST / Non-Orthogonal Multiple Access), multiplexing more than 2 users in the power domain induces excessive error propagation in successive interference cancellation and unacceptable receiver processing latency.

### 3.5 Pairing-Method BER Comparison
- **Outage comparison**: The benchmark outage chart directly plots each evaluator result's `outage_probability_mean` as a percentage. It does not run another benchmark evaluation or derive outage from BER.
- **Modulation and transmission**: The additional BER sweep uses uncoded BPSK with one transmitted bit per active user and block. OMA users transmit in orthogonal half-slot opportunities; paired methods use two-user PD-NOMA superposition with the configured weak/strong power fractions.
- **Channel and noise**: Each test scenario's existing complex path-loss-plus-Rayleigh channel realization is held constant over its blocks and SNR points. Independent circular complex AWGN is added at the receivers. The SNR sweep normalizes noise to the scenario's mean received channel power and applies the existing slot-quality and slot-noise factors.
- **Detector**: Receivers are assumed to have perfect CSI. The weak user's BPSK decision marginalizes over the unknown strong-user bit. The strong user decodes the weak bit and then performs hard SIC; the configured SIC residual factor is retained. OMA uses coherent maximum-likelihood BPSK detection. BER is computed from aggregate bit errors divided by aggregate transmitted bits.
- **Comparison boundary**: This BER experiment isolates the pairing policies under a common PD-NOMA/OMA physical layer. It does not claim to simulate the hybrid scheduler's per-pair SCMA/PD-NOMA mode choice. The existing SCMA QPSK codebooks, MPA detector, and SCMA BER curve remain a separate experiment and are not reused as BER values for other methods. The BER comparison uses the same test channel scenarios across methods, but its transmission model is not the existing rate/outage scheduler.
