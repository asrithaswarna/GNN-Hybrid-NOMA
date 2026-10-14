# Methodology & Engineering Workflow

## 0. ML Syllabus Extension Workflow

In addition to the original GNN/NOMA experiment, Version C provides a small executable ML lifecycle:

```text
Synthetic or validated CSV scenarios
    -> scenario-level split
    -> feature engineering and training-only preprocessing
    -> Ridge / Decision Tree / Random Forest rate regression
    -> held-out metrics and saved artifacts
    -> reusable inference and lightweight monitoring
```

`src/data_pipeline.py` owns input validation, scenario splitting, and simulation-derived user-rate targets. `src/classical_models.py` owns supervised baselines and MAE, MSE, RMSE, and R2. `src/unsupervised.py` provides K-means, DBSCAN, and PCA for exploratory analysis. These classical models predict a continuous simulated user rate; they are not presented as replacements for the GNN pairing task.

The training-serving boundary is explicit: model pipelines fit imputation and standardization on training rows only, save the complete pipeline with joblib, and reuse it during inference. Scenario identifiers and target columns are not model features. The GNN checkpoint remains a PyTorch artifact. `src/monitoring.py` records local prediction counts, invalid inputs, latency, and the last prediction; this is educational monitoring rather than production observability.

## 1. System Architecture & Workflow Pipeline

The complete pipeline of the proposed Graph Neural Network-Based Dynamic User Pairing and Slot Assignment system is illustrated below:

```
+-------------------------------------------------------------+
|               Phase 1: Wireless Channel Simulation          |
|  - BS Placement (0, 0), Annular Cell Radius (20m - 500m)    |
|  - Log-distance Path Loss + Rayleigh Small-Scale Fading     |
|  - Channel Gains g_i = |h_i|^2, Thermal Noise per Slot      |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               Phase 2: Graph Representation                 |
|  - Node Features (6D): Coordinates, Distance, Gains, QoS    |
|  - Edge Features (4D): Distance, Gain Disparity, Corr, Interf|
|  - Fully Connected Graph Topology (Complete or k-NN)        |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               Phase 3: Supervision Oracle                   |
|  - Exhaustive Combinatorial Search (N <= 10, 945 partitions)|
|  - Edmonds' Blossom Max-Weight Matching (N > 10)            |
|  - Ground Truth Pairing Matrix Y & Slot Vector s            |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               Phase 4: Edge-Aware GNN Processing            |
|  - Input Projection & Node Embeddings                       |
|  - Stacked Edge-Conditioned Convolution Layers              |
|  - Bilinear Edge Scoring Head -> Pair Affinity Logits       |
|  - Multi-Task Head -> Slot Allocation Logits                |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               Phase 5: Constraint-Preserving Matching       |
|  - Sigmoid Affinity Predictions A_ij                        |
|  - Greedy Maximum-Weight Matching Post-Processor            |
|  - Strict Guarantees: Disjoint Pairs, No Duplicates, Odd N  |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               Phase 6: Dynamic Slot & Power Allocation      |
|  - Slot Capacity Enforcement (<= 1 Pair per Slot)           |
|  - Power Allocation: Weak (75%), Strong (25%)               |
|  - Imperfect SIC Receiver Evaluation                        |
|  - Benchmark Metrics: Sum-Rate, Jain's Fairness, Outage     |
+-------------------------------------------------------------+
```

---

## 2. Graph Formulation & Feature Engineering

### 2.1 Why Graph Neural Networks for User Pairing?
Conventional multiple access allocation methods rely on Euclidean sorting (e.g., Near-Far channel gain ordering) or heuristic greedy searches. However:
1. Conventional sorting ignores spatial correlation and multi-path interference potential.
2. Combinatorial optimization (e.g., branch-and-bound or exhaustive search) incurs exponential complexity $\mathcal{O}((N-1)!!)$, rendering real-time execution intractable for millisecond cellular scheduling.
3. Standard multi-layer perceptrons (MLPs) lack permutation equivariance; permuting user indices alters output predictions arbitrarily.

A Graph Neural Network (GNN) naturally respects the topology of wireless multi-user systems:
- **Permutation Equivariance**: The affinity between user $i$ and user $j$ is independent of arbitrary indexing orders.
- **Topological Generalization**: The trained model can perform inference on varying user configurations with constant $\mathcal{O}(V + E)$ forward-pass complexity.

### 2.2 Feature Normalization
All node and edge features are explicitly normalized to $[0, 1]$ or zero-mean unit-variance to prevent vanishing/exploding gradients in message passing layers:
- Spatial positions: $x_i / R_{\max}, y_i / R_{\max}, d_i / R_{\max} \in [0, 1]$.
- Channel gains: Standardized in decibels: $\tilde{g}_i = (g_{i, \text{dB}} - \mu_g) / \sigma_g$.
- Inter-user gain disparity: $|g_{i, \text{dB}} - g_{j, \text{dB}}| / 40.0$.

---

## 3. Training Objective & Loss Function

Because optimal user pairs in a network of $N$ users constitute only $N/2$ positive edges out of $N(N-1)$ candidate directed edges, severe positive-negative class imbalance exists.

To address this, we formulate a class-weighted Binary Cross-Entropy (BCE) loss combined with multi-class Cross-Entropy for slot assignment:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pair}} + \lambda_s \mathcal{L}_{\text{slot}}$$

where:
$$\mathcal{L}_{\text{pair}} = -\frac{1}{|\mathcal{E}|} \sum_{(i, j) \in \mathcal{E}, i \neq j} \left[ w_{\text{pos}} Y_{ij} \log(\sigma(\hat{A}_{ij})) + (1 - Y_{ij}) \log(1 - \sigma(\hat{A}_{ij})) \right]$$
with positive class weight $w_{\text{pos}} = N - 2$, and slot classification weight $\lambda_s = 0.3$.

Optimization is carried out using the Adam optimizer with initial learning rate $\eta = 10^{-3}$, weight decay $10^{-4}$, and a `ReduceLROnPlateau` learning-rate scheduler.

---

## 4. Post-Processing & Hard Constraint Enforcement

The raw GNN output provides continuous edge affinity probabilities $\hat{A}_{ij} \in [0, 1]$. In practical wireless systems, these continuous affinities must be mapped to valid, discrete scheduling decisions.

The post-processing algorithm operates as follows:
1. Symmetrize affinity matrix: $\mathbf{A}_{\text{sym}} = \frac{1}{2}(\hat{\mathbf{A}} + \hat{\mathbf{A}}^T)$.
2. Set diagonal entries to $-\infty$ to preclude self-pairing.
3. Sort all unique candidate pairs $(i, j)$ in descending order of predicted affinity.
4. Greedily select the highest-scoring disjoint pairs until all users are paired or available slots are exhausted.
5. In case of an odd user count ($N \pmod 2 \neq 0$), the remaining user is assigned to single-user OFDMA mode.

This deterministic mapping guarantees:
- **Zero Duplicate Users**: No user appears in multiple slots.
- **100% Valid Schedules**: Always conforms to physical layer capacity constraints.
- **Graceful Odd-User Handling**: No crashed runs or undefined array indices.
