# Mathematical Modeling of Hybrid NOMA Networks & Graph Neural Network Architecture

## 1. Introduction & Network Topology
We consider a downlink single-cell cellular network comprising a centralized Base Station (BS) situated at coordinate origin $(0, 0)$ and a set of $N$ active user equipments (UEs) denoted by:
$$\mathcal{U} = \{u_1, u_2, \dots, u_N\}$$

The coverage area is defined as an annular disk:
$$\mathcal{A} = \left\{(x, y) \in \mathbb{R}^2 \;\middle|\; R_{\min} \le \sqrt{x^2 + y^2} \le R_{\max}\right\}$$
where $R_{\min}$ is the inner exclusion radius avoiding near-field singularity, and $R_{\max}$ represents the cell boundary.

The total system bandwidth $B$ is partitioned into $S$ orthogonal resource blocks (or time-frequency slots):
$$\mathcal{S} = \{s_1, s_2, \dots, s_S\}$$
with bandwidth per slot:
$$B_s = \frac{B}{S}$$

Total BS transmit power budget is denoted by $P_{\text{total}}$, yielding power per slot $P_s = P_{\text{total}} / S$ under uniform slot power allocation.

---

## 2. Wireless Propagation & Channel Model

### 2.1 Large-Scale Path Loss
The large-scale path loss $PL(d_i)$ as a function of Euclidean distance $d_i = \sqrt{x_i^2 + y_i^2}$ follows the standard log-distance path loss model:
$$PL(d_i) \,[\text{dB}] = PL_0 + 10 \alpha \log_{10}\left(\frac{\max(d_i, d_0)}{d_0}\right)$$
where:
- $d_0$ is the reference distance ($1.0\text{ m}$).
- $PL_0$ is the free-space path loss at reference distance $d_0$.
- $\alpha$ is the path-loss attenuation exponent ($3.5$ for urban microcells).

The linear path-loss gain is expressed as:
$$g_{PL, i} = 10^{-\frac{PL(d_i)}{10}}$$

### 2.2 Small-Scale Fading
Small-scale multi-path propagation is modeled via frequency-flat Rayleigh fading:
$$h_{\text{fading}, i} \sim \mathcal{CN}(0, \sigma_f^2) = \frac{1}{\sqrt{2}}(X_i + j Y_i), \quad X_i, Y_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$$

### 2.3 Composite Channel Gain
The complex channel coefficient for user $u_i$ is:
$$h_i = \sqrt{g_{PL, i}} \cdot h_{\text{fading}, i}$$
The corresponding instantaneous channel power gain is:
$$g_i = |h_i|^2$$

### 2.4 Thermal Noise
Thermal noise power per resource slot is given by:
$$\sigma^2 = N_0 \cdot F \cdot B_s$$
where:
- $N_0 = 10^{(N_{0, \text{dBm}} - 30)/10}\text{ W/Hz}$ is the thermal noise power spectral density ($-174\text{ dBm/Hz}$).
- $F = 10^{F_{\text{dB}}/10}$ is the receiver noise figure ($5\text{ dB}$).

The instantaneous signal-to-noise ratio (SNR) for user $u_i$ in single-user mode is:
$$\text{SNR}_i = \frac{P_s g_i}{\sigma^2}$$

---

## 3. Power-Domain NOMA (PD-NOMA) Mathematical Model

In PD-NOMA, two paired users $(u_i, u_j)$ share the same resource slot $s$ via superposition coding in the power domain.

### 3.1 User Role Identification & Power Allocation
Let the two users have channel gains $g_w$ and $g_s$ such that:
$$g_w \le g_s$$
Here, $u_w$ is the **weak (cell-edge)** user and $u_s$ is the **strong (near-cell)** user.

The BS transmits the superimposed signal:
$$x = \sqrt{\alpha_w P_s} s_w + \sqrt{\alpha_s P_s} s_s$$
subject to the power allocation constraint:
$$\alpha_w + \alpha_s \le 1, \quad \alpha_w > \alpha_s > 0$$
More transmit power is assigned to the weak user ($\alpha_w = 0.75, \alpha_s = 0.25$) to maintain user fairness.

### 3.2 Received Signal Model
The signals received at $u_w$ and $u_s$ are:
$$y_w = h_w x + n_w = h_w \left(\sqrt{\alpha_w P_s} s_w + \sqrt{\alpha_s P_s} s_s\right) + n_w$$
$$y_s = h_s x + n_s = h_s \left(\sqrt{\alpha_w P_s} s_w + \sqrt{\alpha_s P_s} s_s\right) + n_s$$
where $n_w, n_s \sim \mathcal{CN}(0, \sigma^2)$.

### 3.3 Decoding Order & Successive Interference Cancellation (SIC)

#### 1. Weak User Detection:
The weak user $u_w$ decodes its message $s_w$ directly, treating the strong user's signal $s_s$ as additive co-channel interference:
$$\text{SINR}_w = \frac{\alpha_w P_s g_w}{\alpha_s P_s g_w + \sigma^2}$$
The achievable rate for the weak user is:
$$R_w = B_s \log_2(1 + \text{SINR}_w)$$

#### 2. Strong User Detection (with Imperfect SIC):
The strong user $u_s$ first decodes the weak user's signal $s_w$:
$$\text{SINR}_{s \to w} = \frac{\alpha_w P_s g_s}{\alpha_s P_s g_s + \sigma^2}$$

Since $g_s \ge g_w$, it is guaranteed that $\text{SINR}_{s \to w} \ge \text{SINR}_w$, ensuring that $u_s$ can successfully decode $s_w$.

Upon decoding $s_w$, $u_s$ reconstructs and subtracts $\sqrt{\alpha_w P_s} s_w$ from $y_s$. In realistic transceivers, channel estimation errors and phase jitter induce imperfect SIC, quantified by residual interference factor $\beta_{\text{SIC}} \in [0, 1)$:
$$\text{SINR}_s = \frac{\alpha_s P_s g_s}{\beta_{\text{SIC}} \alpha_w P_s g_s + \sigma^2}$$
where $\beta_{\text{SIC}} = 0$ represents idealized perfect SIC, and $\beta_{\text{SIC}} > 0$ models residual self-interference.

The achievable rate for the strong user is:
$$R_s = B_s \log_2(1 + \text{SINR}_s)$$

#### 3. Pair Sum-Rate:
$$R_{\text{pair}} = R_w + R_s$$

---

## 4. Sparse Code Multiple Access (SCMA) Mathematical Model

SCMA is a codebook-based non-orthogonal multiple access scheme where $J$ user layers are multiplexed across $K$ orthogonal subcarrier resources with overloading factor $\lambda = J/K > 1$.

### 4.1 Factor Graph Representation
The connectivity between $J=6$ users and $K=4$ orthogonal resource elements is governed by the binary indicator matrix $\mathbf{F} \in \{0, 1\}^{4 \times 6}$:
$$\mathbf{F} = \begin{bmatrix}
1 & 1 & 1 & 0 & 0 & 0 \\
1 & 0 & 0 & 1 & 1 & 0 \\
0 & 1 & 0 & 1 & 0 & 1 \\
0 & 0 & 1 & 0 & 1 & 1
\end{bmatrix}$$
- Column degree $d_v = 2$: each user transmits on 2 subcarriers.
- Row degree $d_f = 3$: each subcarrier superimposes signals from 3 users.

### 4.2 Multi-Dimensional Codebook Mapping
Each user $j \in \{1, \dots, J\}$ maps an input $\log_2(M)$-bit word to a $K$-dimensional complex codeword $\mathbf{c}_j(m) \in \mathcal{C}_j \subset \mathbb{C}^K$ with cardinality $M=4$:
$$\mathbf{c}_j(m) = \begin{bmatrix} c_{1, j}(m) \\ \vdots \\ c_{K, j}(m) \end{bmatrix}, \quad \text{where } c_{k, j}(m) = 0 \iff F_{k, j} = 0$$

### 4.3 Received Signal Vector
$$\mathbf{y} = \sum_{j=1}^J \operatorname{diag}(\mathbf{h}_j) \mathbf{c}_j(m_j) + \mathbf{w}$$
where $\mathbf{w} \sim \mathcal{CN}(\mathbf{0}, \sigma^2 \mathbf{I}_K)$.

### 4.4 Iterative Message Passing Algorithm (MPA)
The MPA performs belief propagation over the factor graph through iterative message passing:
1. **Resource-to-User Probability Message**:
   $$I_{f_k \to v_j}(m) = \sum_{\sim \{m_j\}} \exp\left(-\frac{1}{\sigma^2} \left| y_k - \sum_{j' \in \mathcal{V}_k} h_{k, j'} c_{k, j'}(m_{j'}) \right|^2 \right) \prod_{j' \in \mathcal{V}_k \setminus \{j\}} I_{v_{j'} \to f_k}(m_{j'})$$
2. **User-to-Resource Probability Message**:
   $$I_{v_j \to f_k}(m) = \frac{1}{M} \prod_{k' \in \mathcal{K}_j \setminus \{k\}} I_{f_{k'} \to v_j}(m)$$
3. **Symbol Posterior Probability & Decision Rule**:
   $$Q_j(m) = \frac{1}{M} \prod_{k \in \mathcal{K}_j} I_{f_k \to v_j}(m)$$
   $$\hat{m}_j = \arg\max_{m \in \{0, \dots, M-1\}} Q_j(m)$$

---

## 5. Network Optimization Formulation

The dynamic user pairing and slot assignment problem is formulated as a combinatorial binary integer program.

### 5.1 Decision Variables
- Binary pairing indicator $X_{ij} \in \{0, 1\}$: equals $1$ if user $u_i$ is paired with user $u_j$, $0$ otherwise.
- Binary slot allocation indicator $S_{ik} \in \{0, 1\}$: equals $1$ if user $u_i$ is assigned to slot $k$.

### 5.2 Objective Function
Maximize the combined utility of network sum-rate and Jain's fairness index:
$$\max_{\mathbf{X}, \mathbf{S}} \quad \mathcal{U}(\mathbf{X}, \mathbf{S}) = \sum_{k=1}^S R_k(\mathbf{X}, \mathbf{S}) + \lambda_{\text{fair}} \cdot J(\mathbf{R})$$
where:
$$J(\mathbf{R}) = \frac{\left(\sum_{i=1}^N R_i\right)^2}{N \sum_{i=1}^N R_i^2}$$

### 5.3 Constraints
1. **Disjoint Pairing Constraint**: Each user is paired at most once:
   $$\sum_{j=1, j \neq i}^N X_{ij} \le 1, \quad \forall i \in \{1, \dots, N\}$$
2. **Symmetry & Irreflexivity**:
   $$X_{ij} = X_{ji}, \quad X_{ii} = 0, \quad \forall i, j$$
3. **Slot Capacity Constraint**: Each slot hosts at most one user pair (or one single user):
   $$\sum_{i=1}^N S_{ik} \le 2, \quad \forall k \in \{1, \dots, S\}$$
4. **Power Budget**:
   $$\sum_{k=1}^S P_k \le P_{\text{total}}$$
5. **Quality of Service (QoS) Minimum Rate Constraint**:
   $$R_i \ge R_{\min}, \quad \forall i \in \{1, \dots, N\}$$

---

## 6. Graph Neural Network (GNN) Formulation

### 6.1 Graph Representation $\mathcal{G} = (\mathcal{V}, \mathcal{E})$
- **Nodes $\mathcal{V}$**: $N$ users.
- **Node Feature Vector $\mathbf{x}_i \in \mathbb{R}^6$**:
  $$\mathbf{x}_i = \left[ \frac{x_i}{R_{\max}}, \frac{y_i}{R_{\max}}, \frac{d_i}{R_{\max}}, \frac{g_{i, \text{dB}} - \mu_g}{\sigma_g}, \frac{g_i}{\bar{g}}, \frac{R_{\min, i}}{R_{\text{ref}}} \right]^T$$
- **Edge Feature Vector $\mathbf{e}_{ij} \in \mathbb{R}^4$**:
  $$\mathbf{e}_{ij} = \left[ \frac{\|p_i - p_j\|}{2 R_{\max}}, \frac{|g_{i, \text{dB}} - g_{j, \text{dB}}|}{40}, \frac{|\mathbf{h}_i^H \mathbf{h}_j|}{\|\mathbf{h}_i\| \|\mathbf{h}_j\|}, \frac{g_i + g_j}{2 \bar{g}} \right]^T$$

### 6.2 Edge-Conditioned Message Passing Layer
At layer $l \in \{1, \dots, L\}$:
$$\mathbf{m}_{j \to i}^{(l)} = \text{MLP}_{\text{msg}}^{(l)}\left(\left[\mathbf{h}_j^{(l-1)}, \mathbf{e}_{ij}\right]\right)$$
$$\mathbf{a}_i^{(l)} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{j \to i}^{(l)}$$
$$\mathbf{h}_i^{(l)} = \text{LayerNorm}\left(\mathbf{h}_i^{(l-1)} + \text{MLP}_{\text{upd}}^{(l)}\left(\left[\mathbf{h}_i^{(l-1)}, \mathbf{a}_i^{(l)}\right]\right)\right)$$

### 6.3 Pair Affinity Prediction & Post-Processing
$$\hat{A}_{ij} = \sigma\left(\text{MLP}_{\text{edge}}\left(\left[\mathbf{h}_i^{(L)}, \mathbf{h}_j^{(L)}, |\mathbf{h}_i^{(L)} - \mathbf{h}_j^{(L)}|, \mathbf{h}_i^{(L)} \odot \mathbf{h}_j^{(L)}\right]\right)\right)$$
Continuous affinity $\hat{\mathbf{A}}$ is converted into discrete disjoint pairs via maximum-weight matching:
$$\mathbf{X}^* = \arg\max_{\mathbf{X} \in \mathcal{M}} \sum_{i < j} \hat{A}_{ij} X_{ij}$$
guaranteeing constraint satisfaction and zero invalid assignments.
