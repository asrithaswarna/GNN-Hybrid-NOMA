"""
Power-Domain Non-Orthogonal Multiple Access (PD-NOMA) module.
Implements 2-user superposition coding, successive interference cancellation (SIC),
SINR analysis with imperfect SIC, and achievable rate computation.
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np
from src.config import PDNOMAConfig
from src.channel_model import UserChannelState


@dataclass
class PDPairResult:
    """Detailed transmission metrics for a 2-user PD-NOMA pair in a slot."""
    weak_user_id: int
    strong_user_id: int
    slot_id: int
    power_w: float
    bandwidth_hz: float
    noise_power_w: float
    alpha_weak: float
    alpha_strong: float
    sic_residual_factor: float
    gain_weak: float
    gain_strong: float
    sinr_weak: float
    sinr_strong_to_weak: float
    sinr_strong: float
    rate_weak_bps: float
    rate_strong_bps: float
    pair_sum_rate_bps: float
    sic_successful: bool
    weak_outage: bool
    strong_outage: bool


class PDNOMAEngine:
    """
    Computes SINRs, achievable rates, and decoding orders for PD-NOMA pairs.
    """

    def __init__(self, cfg: PDNOMAConfig):
        self.cfg = cfg

    def compute_pair_rates(
        self,
        u_a: UserChannelState,
        u_b: UserChannelState,
        power_w: float,
        bandwidth_hz: float,
        noise_power_w: float,
        slot_id: int = 0,
        custom_alpha_weak: Optional[float] = None,
    ) -> PDPairResult:
        """
        Calculate achievable rates for two users multiplexed on the same slot using PD-NOMA.
        Automatically orders users by channel gain: weak user has smaller gain.
        """
        # Identify weak and strong users
        if u_a.channel_gain <= u_b.channel_gain:
            weak_u, strong_u = u_a, u_b
        else:
            weak_u, strong_u = u_b, u_a

        g_w = float(weak_u.channel_gain)
        g_s = float(strong_u.channel_gain)

        a_w = custom_alpha_weak if custom_alpha_weak is not None else self.cfg.alpha_weak
        a_s = 1.0 - a_w
        beta_sic = self.cfg.sic_residual_factor

        p = power_w
        sigma_sq = noise_power_w

        # 1. Weak user decodes its own signal s_w treating s_s as interference
        denom_w = a_s * p * g_w + sigma_sq
        sinr_w = (a_w * p * g_w) / denom_w if denom_w > 0 else 0.0
        rate_w = bandwidth_hz * np.log2(1.0 + sinr_w)

        # 2. Strong user decodes weak user's signal s_w first
        denom_s_to_w = a_s * p * g_s + sigma_sq
        sinr_s_to_w = (a_w * p * g_s) / denom_s_to_w if denom_s_to_w > 0 else 0.0
        rate_s_to_w = bandwidth_hz * np.log2(1.0 + sinr_s_to_w)

        # Condition for SIC feasibility: rate_s_to_w >= rate_w (always holds when g_s >= g_w)
        sic_successful = bool(rate_s_to_w >= rate_w - 1e-7)

        # 3. Strong user subtracts s_w with residual factor beta_sic, then decodes s_s
        denom_s = beta_sic * a_w * p * g_s + sigma_sq
        sinr_s = (a_s * p * g_s) / denom_s if denom_s > 0 else 0.0
        rate_s = bandwidth_hz * np.log2(1.0 + sinr_s)

        # Outage check against user QoS requirements
        weak_outage = bool(rate_w < weak_u.min_rate_req_bps)
        strong_outage = bool(rate_s < strong_u.min_rate_req_bps)

        return PDPairResult(
            weak_user_id=weak_u.user_id,
            strong_user_id=strong_u.user_id,
            slot_id=slot_id,
            power_w=power_w,
            bandwidth_hz=bandwidth_hz,
            noise_power_w=noise_power_w,
            alpha_weak=a_w,
            alpha_strong=a_s,
            sic_residual_factor=beta_sic,
            gain_weak=g_w,
            gain_strong=g_s,
            sinr_weak=float(sinr_w),
            sinr_strong_to_weak=float(sinr_s_to_w),
            sinr_strong=float(sinr_s),
            rate_weak_bps=float(rate_w),
            rate_strong_bps=float(rate_s),
            pair_sum_rate_bps=float(rate_w + rate_s),
            sic_successful=sic_successful,
            weak_outage=weak_outage,
            strong_outage=strong_outage,
        )

    def compute_single_user_oma_rate(
        self,
        user: UserChannelState,
        power_w: float,
        bandwidth_hz: float,
        noise_power_w: float,
    ) -> float:
        """Calculate OFDMA (single user in slot) benchmark rate."""
        snr = (power_w * user.channel_gain) / noise_power_w if noise_power_w > 0 else 0.0
        return float(bandwidth_hz * np.log2(1.0 + snr))


def compute_jains_fairness_index(rates: List[float]) -> float:
    """
    Jain's Fairness Index:
    J = (sum(R_i))^2 / (N * sum(R_i^2))
    Bounded in [1/N, 1.0].
    """
    arr = np.array(rates, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0
    sum_r = float(np.sum(arr))
    sum_sq_r = float(np.sum(arr ** 2))
    if sum_sq_r == 0.0:
        return 1.0  # Equal zero rates is technically fair
    return float((sum_r ** 2) / (n * sum_sq_r))
