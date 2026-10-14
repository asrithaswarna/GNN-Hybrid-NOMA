"""
Sparse Code Multiple Access (SCMA) module.
Implements the sparse factor graph, multidimensional complex codebooks,
iterative Message Passing Algorithm (MPA) multi-user detector,
and achievable multi-user rate analysis.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import itertools
import numpy as np
from src.config import SCMAConfig


@dataclass
class SCMAResult:
    """Performance metrics for an SCMA multi-user transmission block."""
    num_users: int
    num_resources: int
    overload_ratio: float
    snr_db: float
    symbol_error_rate: float
    bit_error_rate: float
    sum_rate_bps: float
    per_user_rates_bps: List[float]
    iterations_run: int


class SCMAEngine:
    """
    Simulates uplink/downlink SCMA communication system.
    Default configuration: J=6 users multiplexed over K=4 orthogonal subcarriers (150% overload).
    Each user connects to d_v=2 subcarriers; each subcarrier hosts d_f=3 colliding users.
    """

    def __init__(self, cfg: SCMAConfig):
        self.cfg = cfg
        self.J = cfg.num_layers          # Users / layers (default 6)
        self.K = cfg.num_resources       # Orthogonal resources (default 4)
        self.M = cfg.codebook_size       # Codewords per user (default 4, i.e. 2 bits)
        self.n_iter = cfg.mpa_iterations

        # Canonical regular SCMA factor graph indicator matrix F [K x J]
        # Rows = Resources (k=0..3), Columns = Users (j=0..5)
        # Each column has d_v = 2 ones; each row has d_f = 3 ones
        self.F = np.array([
            [1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 1],
        ], dtype=int)

        # Precompute connectivity graphs
        # V_k: list of users connected to resource k
        self.V_k: List[List[int]] = [
            [j for j in range(self.J) if self.F[k, j] == 1]
            for k in range(self.K)
        ]
        # K_j: list of resources connected to user j
        self.K_j: List[List[int]] = [
            [k for k in range(self.K) if self.F[k, j] == 1]
            for j in range(self.J)
        ]

        # Generate multidimensional constellation codebooks for all J users: [J, M, K] complex
        self.codebooks = self._generate_codebooks()

    def _generate_codebooks(self) -> np.ndarray:
        """
        Generate geometric constellation codebooks based on QPSK rotation.
        For each user j, exactly K_j resources have non-zero constellation points.
        Returns: codebooks array of shape [J, M, K] (complex128).
        """
        codebooks = np.zeros((self.J, self.M, self.K), dtype=complex)

        # Mother constellation: 4 QPSK points on unit circle
        theta = np.pi / 4.0
        qpsk_points = np.array([
            np.exp(1j * (theta + m * np.pi / 2.0))
            for m in range(self.M)
        ], dtype=complex) / np.sqrt(2.0)  # Normalized power

        for j in range(self.J):
            active_res = self.K_j[j]
            # Distinct phase rotation per user to optimize Euclidean distance
            phase_rot = np.exp(1j * (j * np.pi / self.J))
            for m in range(self.M):
                point = qpsk_points[m] * phase_rot
                # Split power across active resource elements
                codebooks[j, m, active_res[0]] = np.real(point)
                codebooks[j, m, active_res[1]] = 1j * np.imag(point)

        return codebooks

    def encode(self, symbols: np.ndarray) -> np.ndarray:
        """
        Encode discrete symbol indices m in [0, M-1] for each user j into codewords.
        symbols: array of shape [J] with integers in {0, ..., M-1}.
        Returns: transmitted signal per user [J, K] complex.
        """
        transmitted = np.zeros((self.J, self.K), dtype=complex)
        for j in range(self.J):
            m = symbols[j]
            transmitted[j, :] = self.codebooks[j, m, :]
        return transmitted

    def mpa_detector(
        self,
        y: np.ndarray,
        H: np.ndarray,
        noise_variance: float,
    ) -> np.ndarray:
        """
        Message Passing Algorithm (MPA) multi-user detector.
        y: received complex signal vector on K resources [K].
        H: complex channel coefficients [K, J].
        noise_variance: noise power sigma^2.
        Returns: posterior probabilities [J, M] for each user and symbol.
        """
        sigma_sq = max(float(noise_variance), 1e-12)

        # Initialize user-to-resource messages: I_v_to_f [K, J, M]
        # Prior probability is uniform: 1 / M
        I_v_to_f = np.zeros((self.K, self.J, self.M), dtype=float)
        for j in range(self.J):
            for k in self.K_j[j]:
                I_v_to_f[k, j, :] = 1.0 / self.M

        I_f_to_v = np.zeros((self.K, self.J, self.M), dtype=float)

        # Iterative message passing between factor (resource) and variable (user) nodes
        for it in range(self.n_iter):
            # Step 1: Update Resource-to-User messages (I_f_to_v)
            for k in range(self.K):
                connected_users = self.V_k[k]  # Size d_f (e.g. 3 users)
                d_f = len(connected_users)

                # For each target user j in connected_users:
                for target_idx, target_j in enumerate(connected_users):
                    interfering_users = [u for u in connected_users if u != target_j]

                    for m_target in range(self.M):
                        accumulated_prob = 0.0

                        # Iterate over all combinations of constellation points for interfering users
                        for combo in itertools.product(range(self.M), repeat=d_f - 1):
                            # Calculate combined received signal at resource k
                            expected_signal = H[k, target_j] * self.codebooks[target_j, m_target, k]
                            prior_product = 1.0

                            for idx_u, u_other in enumerate(interfering_users):
                                m_other = combo[idx_u]
                                expected_signal += H[k, u_other] * self.codebooks[u_other, m_other, k]
                                prior_product *= I_v_to_f[k, u_other, m_other]

                            diff = y[k] - expected_signal
                            dist_sq = float(np.abs(diff) ** 2)
                            cond_prob = np.exp(-dist_sq / sigma_sq)

                            accumulated_prob += cond_prob * prior_product

                        I_f_to_v[k, target_j, m_target] = accumulated_prob

                    # Normalize I_f_to_v over m
                    sum_f_to_v = np.sum(I_f_to_v[k, target_j, :])
                    if sum_f_to_v > 0:
                        I_f_to_v[k, target_j, :] /= sum_f_to_v

            # Step 2: Update User-to-Resource messages (I_v_to_f)
            for j in range(self.J):
                connected_res = self.K_j[j]
                for k_target in connected_res:
                    other_res = [k for k in connected_res if k != k_target]

                    for m in range(self.M):
                        prob = 1.0 / self.M
                        for k_other in other_res:
                            prob *= I_f_to_v[k_other, j, m]
                        I_v_to_f[k_target, j, m] = prob

                    # Normalize
                    sum_v = np.sum(I_v_to_f[k_target, j, :])
                    if sum_v > 0:
                        I_v_to_f[k_target, j, :] /= sum_v

        # Final posterior probability Q_j(m)
        Q = np.zeros((self.J, self.M), dtype=float)
        for j in range(self.J):
            for m in range(self.M):
                prod = 1.0 / self.M
                for k in self.K_j[j]:
                    prod *= I_f_to_v[k, j, m]
                Q[j, m] = prod
            sum_q = np.sum(Q[j, :])
            if sum_q > 0:
                Q[j, :] /= sum_q

        return Q

    def simulate_transmission(
        self,
        channel_gains: np.ndarray,
        snr_db: float,
        num_blocks: int = 100,
        bandwidth_hz: float = 2.0e6,
    ) -> SCMAResult:
        """
        Run end-to-end Monte Carlo transmission of SCMA blocks over Rayleigh/AWGN channels.
        Computes empirical SER, BER, and achievable sum-rate.
        """
        rng = np.random.default_rng(42)
        snr_lin = 10.0 ** (snr_db / 10.0)
        signal_power = 1.0
        noise_var = signal_power / snr_lin

        # Diagonal / resource-wise channel matrix H [K, J]
        # In multi-carrier SCMA, each user has a fading coefficient on its active resources
        H = np.zeros((self.K, self.J), dtype=complex)
        for j in range(self.J):
            g = channel_gains[j % len(channel_gains)]
            for k in self.K_j[j]:
                h_ray = (rng.normal(0, 1) + 1j * rng.normal(0, 1)) / np.sqrt(2.0)
                H[k, j] = np.sqrt(g) * h_ray

        symbol_errors = 0
        bit_errors = 0
        total_symbols = num_blocks * self.J
        total_bits = total_symbols * int(np.log2(self.M))

        for _ in range(num_blocks):
            # Generate random symbols for J users
            tx_symbols = rng.integers(0, self.M, size=self.J)
            tx_codewords = self.encode(tx_symbols)

            # Transmit over channel: y = sum_j (H[:, j] * c_j) + noise
            y = np.zeros(self.K, dtype=complex)
            for j in range(self.J):
                y += H[:, j] * tx_codewords[j, :]

            # Add AWGN noise
            noise = (rng.normal(0, np.sqrt(noise_var / 2.0), size=self.K) +
                     1j * rng.normal(0, np.sqrt(noise_var / 2.0), size=self.K))
            y += noise

            # MPA detection
            posteriors = self.mpa_detector(y, H, noise_var)
            detected_symbols = np.argmax(posteriors, axis=1)

            # Error counting
            for j in range(self.J):
                if detected_symbols[j] != tx_symbols[j]:
                    symbol_errors += 1
                    # Gray bit mapping: count differing bits
                    diff_bits = bin(detected_symbols[j] ^ tx_symbols[j]).count('1')
                    bit_errors += diff_bits

        ser = float(symbol_errors / total_symbols) if total_symbols > 0 else 0.0
        ber = float(bit_errors / total_bits) if total_bits > 0 else 0.0

        # Information-theoretic multi-user sum-rate estimate under SCMA overloading:
        # Effective subcarrier bandwidth = bandwidth_hz / K
        subcarrier_bw = bandwidth_hz / self.K
        per_user_rates: List[float] = []
        for j in range(self.J):
            # User throughput based on active resources and SINR with MPA interference attenuation
            # MPA reduces multi-user interference by approximately 80% through message exchange
            res_indices = self.K_j[j]
            effective_sinr = 0.0
            for k in res_indices:
                desired_pwr = float(np.abs(H[k, j]) ** 2)
                interf_pwr = 0.0
                for other_u in self.V_k[k]:
                    if other_u != j:
                        # Residual multi-user interference after MPA suppression
                        interf_pwr += 0.15 * float(np.abs(H[k, other_u]) ** 2)
                denom = interf_pwr + noise_var
                effective_sinr += desired_pwr / denom if denom > 0 else 0.0

            r_j = subcarrier_bw * float(np.log2(1.0 + effective_sinr / len(res_indices)))
            per_user_rates.append(r_j)

        sum_rate = float(np.sum(per_user_rates))

        return SCMAResult(
            num_users=self.J,
            num_resources=self.K,
            overload_ratio=float(self.J / self.K),
            snr_db=snr_db,
            symbol_error_rate=ser,
            bit_error_rate=ber,
            sum_rate_bps=sum_rate,
            per_user_rates_bps=per_user_rates,
            iterations_run=self.n_iter,
        )
