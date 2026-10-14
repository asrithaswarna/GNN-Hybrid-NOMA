"""
Wireless channel and network simulation module.
Models cellular geometry, distance-dependent path loss, Rayleigh fading, and noise.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from src.config import NetworkConfig, ChannelConfig


@dataclass
class UserChannelState:
    """Represents a single user's spatial location and physical channel realization."""
    user_id: int
    x: float
    y: float
    distance: float
    path_loss_db: float
    path_loss_linear: float
    fading_complex: complex
    channel_complex: complex
    channel_gain: float         # |h|^2
    channel_gain_db: float      # 10 * log10(|h|^2)
    snr_linear: float
    snr_db: float
    min_rate_req_bps: float


class WirelessNetwork:
    """
    Simulates a single-cell wireless network with a Base Station (BS) at the origin
    and N distributed User Equipments (UEs).
    """

    def __init__(
        self,
        net_cfg: NetworkConfig,
        chan_cfg: ChannelConfig,
        seed: Optional[int] = None,
    ):
        self.net_cfg = net_cfg
        self.chan_cfg = chan_cfg
        self.rng = np.random.default_rng(seed)

        # Precompute thermal noise power per slot
        # N0_W_Hz: convert dBm/Hz to W/Hz -> 10^((dBm - 30) / 10)
        n0_w_hz = 10.0 ** ((self.net_cfg.noise_spectral_density_dbm - 30.0) / 10.0)
        noise_figure_linear = 10.0 ** (self.net_cfg.noise_figure_db / 10.0)
        self.bandwidth_per_slot = self.net_cfg.total_bandwidth / self.net_cfg.num_slots
        self.noise_power_per_slot = n0_w_hz * noise_figure_linear * self.bandwidth_per_slot
        self.power_per_slot = self.net_cfg.total_transmit_power_w / self.net_cfg.num_slots

    def set_seed(self, seed: int) -> None:
        """Update RNG seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def calculate_path_loss_db(self, distance: np.ndarray) -> np.ndarray:
        """
        Log-distance path loss model:
        PL(d) [dB] = PL_0 + 10 * alpha * log10(d / d_0)
        """
        d_clipped = np.maximum(distance, self.chan_cfg.reference_distance)
        pl_db = (
            self.chan_cfg.reference_path_loss_db
            + 10.0 * self.chan_cfg.path_loss_exponent * np.log10(d_clipped / self.chan_cfg.reference_distance)
        )
        return pl_db

    def generate_users(
        self,
        num_users: Optional[int] = None,
        min_rate_bps: float = 1.0e5,
    ) -> List[UserChannelState]:
        """
        Generate N users with uniform spatial distribution in an annular cell
        and independent Rayleigh fading realizations.
        """
        n = num_users if num_users is not None else self.net_cfg.num_users

        # Area-uniform radial distribution between R_min and R_max
        r_min_sq = self.net_cfg.cell_radius_min ** 2
        r_max_sq = self.net_cfg.cell_radius_max ** 2
        u = self.rng.uniform(0.0, 1.0, size=n)
        radii = np.sqrt(u * (r_max_sq - r_min_sq) + r_min_sq)

        # Uniform angular distribution [0, 2*pi)
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=n)
        x_coords = radii * np.cos(angles)
        y_coords = radii * np.sin(angles)

        # Distance from Base Station at (0, 0)
        distances = radii

        # Path loss calculation
        pl_db = self.calculate_path_loss_db(distances)
        pl_linear = 10.0 ** (-pl_db / 10.0)

        # Small-scale Rayleigh fading: CN(0, sigma_fading^2)
        # Re and Im are independent N(0, sigma_fading^2 / 2)
        sigma = np.sqrt(self.chan_cfg.fading_variance / 2.0)
        h_real = self.rng.normal(0.0, sigma, size=n)
        h_imag = self.rng.normal(0.0, sigma, size=n)
        h_fading = h_real + 1j * h_imag

        # Combined channel coefficient h = sqrt(PL) * h_fading
        h_combined = np.sqrt(pl_linear) * h_fading
        channel_gains = np.abs(h_combined) ** 2

        users: List[UserChannelState] = []
        for i in range(n):
            gain = float(channel_gains[i])
            gain_db = 10.0 * np.log10(gain) if gain > 1e-30 else -300.0
            snr_lin = float((self.power_per_slot * gain) / self.noise_power_per_slot)
            snr_db = 10.0 * np.log10(snr_lin) if snr_lin > 1e-30 else -300.0

            users.append(
                UserChannelState(
                    user_id=i,
                    x=float(x_coords[i]),
                    y=float(y_coords[i]),
                    distance=float(distances[i]),
                    path_loss_db=float(pl_db[i]),
                    path_loss_linear=float(pl_linear[i]),
                    fading_complex=complex(h_fading[i]),
                    channel_complex=complex(h_combined[i]),
                    channel_gain=gain,
                    channel_gain_db=gain_db,
                    snr_linear=snr_lin,
                    snr_db=snr_db,
                    min_rate_req_bps=min_rate_bps,
                )
            )

        return users
