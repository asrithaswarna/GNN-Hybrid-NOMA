"""
Slot assignment and resource scheduling module for Hybrid NOMA networks.
Maps paired users and unpaired single users to orthogonal time/frequency slots,
strictly enforcing slot capacity, power budget, and mode selection.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from src.channel_model import UserChannelState
from src.noma_pd import PDNOMAEngine, PDPairResult
from src.pairing import PairingPlan, UserPair


def _as_numpy_slot_logits(slot_logits: Any) -> np.ndarray:
    """Normalize slot logits into a numpy array shaped [N, num_slots]."""
    if hasattr(slot_logits, "detach"):
        slot_logits = slot_logits.detach().cpu().numpy()
    arr = np.asarray(slot_logits, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


@dataclass
class SlotTransmissionRecord:
    """Detailed record of transmission in a single resource slot."""
    slot_id: int
    mode: str                    # "PD-NOMA", "OMA_SINGLE", or "IDLE"
    assigned_user_ids: List[int]
    power_w: float
    bandwidth_hz: float
    slot_sum_rate_bps: float
    per_user_rates_bps: Dict[int, float]
    pd_pair_details: Optional[PDPairResult] = None


@dataclass
class NetworkScheduleResult:
    """Full network transmission outcome across all S slots."""
    slots: List[SlotTransmissionRecord]
    total_sum_rate_bps: float
    per_user_rates_bps: Dict[int, float]
    jains_fairness: float
    active_slots: int
    total_slots: int
    slot_utilization: float
    outage_user_count: int
    outage_probability: float


class SlotAssignmentEngine:
    """
    Manages slot assignment, hybrid PD-NOMA/SCMA mode selection, and power allocation
    for paired and unpaired users while respecting slot capacity and deterministic
    conflict resolution.
    """

    def __init__(
        self,
        num_slots: int,
        total_bandwidth_hz: float,
        total_power_w: float,
        noise_power_per_slot: float,
        pd_engine: PDNOMAEngine,
        gain_ratio_threshold: float = 1.5,
    ):
        self.num_slots = num_slots
        self.total_bandwidth = total_bandwidth_hz
        self.total_power = total_power_w
        self.noise_power_per_slot = noise_power_per_slot
        self.pd_engine = pd_engine
        self.gain_ratio_threshold = gain_ratio_threshold

        self.slot_bandwidth = total_bandwidth_hz / max(num_slots, 1)
        self.slot_power = total_power_w / max(num_slots, 1)

    def _slot_quality_factor(self, slot_id: int) -> float:
        """A deterministic, reproducible slot condition affecting channel quality."""
        if self.num_slots <= 1:
            return 1.0
        phase = (slot_id + 1) * np.pi / self.num_slots
        return 0.7 + 0.45 * np.sin(phase + 0.45)

    def _slot_noise_factor(self, slot_id: int) -> float:
        """Deterministic noise/interference factor tied to slot id."""
        if self.num_slots <= 1:
            return 1.0
        phase = (slot_id + 1) * np.pi / self.num_slots
        return 1.0 + 0.35 * np.cos(phase + 1.1)

    def _compute_effective_pair_rates(
        self,
        u_a: UserChannelState,
        u_b: UserChannelState,
        slot_id: int,
        power_w: Optional[float] = None,
        bandwidth_hz: Optional[float] = None,
        noise_power_w: Optional[float] = None,
    ) -> PDPairResult:
        slot_gain = self._slot_quality_factor(slot_id)
        slot_noise = self._slot_noise_factor(slot_id)
        eff_power = power_w if power_w is not None else self.slot_power
        eff_bw = bandwidth_hz if bandwidth_hz is not None else self.slot_bandwidth
        eff_noise = noise_power_w if noise_power_w is not None else self.noise_power_per_slot * slot_noise

        u_a_eff = UserChannelState(
            user_id=u_a.user_id,
            x=u_a.x,
            y=u_a.y,
            distance=u_a.distance,
            path_loss_db=u_a.path_loss_db,
            path_loss_linear=u_a.path_loss_linear,
            fading_complex=u_a.fading_complex,
            channel_complex=u_a.channel_complex * np.sqrt(slot_gain),
            channel_gain=u_a.channel_gain * slot_gain,
            channel_gain_db=10.0 * np.log10(max(u_a.channel_gain * slot_gain, 1e-30)),
            snr_linear=(eff_power * (u_a.channel_gain * slot_gain)) / max(eff_noise, 1e-30),
            snr_db=0.0,
            min_rate_req_bps=u_a.min_rate_req_bps,
        )
        u_b_eff = UserChannelState(
            user_id=u_b.user_id,
            x=u_b.x,
            y=u_b.y,
            distance=u_b.distance,
            path_loss_db=u_b.path_loss_db,
            path_loss_linear=u_b.path_loss_linear,
            fading_complex=u_b.fading_complex,
            channel_complex=u_b.channel_complex * np.sqrt(slot_gain),
            channel_gain=u_b.channel_gain * slot_gain,
            channel_gain_db=10.0 * np.log10(max(u_b.channel_gain * slot_gain, 1e-30)),
            snr_linear=(eff_power * (u_b.channel_gain * slot_gain)) / max(eff_noise, 1e-30),
            snr_db=0.0,
            min_rate_req_bps=u_b.min_rate_req_bps,
        )
        u_a_eff.snr_db = 10.0 * np.log10(max(u_a_eff.snr_linear, 1e-30))
        u_b_eff.snr_db = 10.0 * np.log10(max(u_b_eff.snr_linear, 1e-30))

        return self.pd_engine.compute_pair_rates(
            u_a_eff,
            u_b_eff,
            power_w=eff_power,
            bandwidth_hz=eff_bw,
            noise_power_w=eff_noise,
            slot_id=slot_id,
        )

    def _compute_scma_pair_rate(
        self,
        u_a: UserChannelState,
        u_b: UserChannelState,
        slot_id: int,
        power_w: Optional[float] = None,
        bandwidth_hz: Optional[float] = None,
        noise_power_w: Optional[float] = None,
    ) -> float:
        """A physically-motivated SCMA-style rate estimate for a 2-user allocation."""
        slot_gain = self._slot_quality_factor(slot_id)
        slot_noise = self._slot_noise_factor(slot_id)
        eff_power = power_w if power_w is not None else self.slot_power
        eff_bw = bandwidth_hz if bandwidth_hz is not None else self.slot_bandwidth
        eff_noise = noise_power_w if noise_power_w is not None else self.noise_power_per_slot * slot_noise

        g_a = max(u_a.channel_gain * slot_gain, 1e-30)
        g_b = max(u_b.channel_gain * slot_gain, 1e-30)
        interference = 0.18 * (g_a + g_b) + eff_noise
        sinr_pair = (eff_power * (g_a + g_b)) / max(interference, 1e-30)
        return float(eff_bw * np.log2(1.0 + sinr_pair / 2.0))

    def select_communication_mode(
        self,
        u_a: UserChannelState,
        u_b: UserChannelState,
        slot_id: int,
    ) -> str:
        """Explicit deterministic rule: choose PD-NOMA when it is better and feasible."""
        pd_res = self._compute_effective_pair_rates(u_a, u_b, slot_id)
        scma_rate = self._compute_scma_pair_rate(u_a, u_b, slot_id)
        gain_ratio = max(u_a.channel_gain, u_b.channel_gain) / max(min(u_a.channel_gain, u_b.channel_gain), 1e-30)
        if gain_ratio >= self.gain_ratio_threshold and pd_res.pair_sum_rate_bps >= scma_rate:
            return "PD-NOMA"
        if scma_rate > pd_res.pair_sum_rate_bps:
            return "SCMA"
        return "PD-NOMA"

    def resolve_gnn_slot_mapping(
        self,
        pairing_plan: PairingPlan,
        users: List[UserChannelState],
        slot_logits: Any,
    ) -> List[int]:
        """Resolve GNN slot logits into a deterministic, capacity-valid pair-to-slot mapping."""
        arr = _as_numpy_slot_logits(slot_logits)
        if arr.shape[0] != len(users):
            arr = np.zeros((len(users), self.num_slots), dtype=np.float64)
            for i in range(len(users)):
                arr[i, :] = np.linspace(0.0, 1.0, self.num_slots)

        user_index = {u.user_id: i for i, u in enumerate(users)}
        mapping: List[int] = [0] * len(pairing_plan.pairs)
        occupied: set[int] = set()

        pair_order = sorted(
            range(len(pairing_plan.pairs)),
            key=lambda idx: (
                float(np.max(np.mean(arr[[user_index[pairing_plan.pairs[idx].user1_id], user_index[pairing_plan.pairs[idx].user2_id]], :], axis=0))),
                -idx,
            ),
            reverse=True,
        )

        for pair_idx in pair_order:
            pair = pairing_plan.pairs[pair_idx]
            u1 = user_index[pair.user1_id]
            u2 = user_index[pair.user2_id]
            slot_pref = np.mean(arr[[u1, u2], :], axis=0)
            ranked_slots = np.argsort(slot_pref)[::-1].tolist()
            chosen = None
            for slot in ranked_slots:
                if slot not in occupied:
                    chosen = int(slot)
                    break
            if chosen is None:
                for slot in range(self.num_slots):
                    if slot not in occupied:
                        chosen = slot
                        break
            if chosen is None:
                chosen = pair_idx % self.num_slots
            mapping[pair_idx] = chosen
            occupied.add(chosen)

        return mapping

    def evaluate_schedule(
        self,
        pairing_plan: PairingPlan,
        users: List[UserChannelState],
        custom_slot_mapping: Optional[List[int]] = None,
    ) -> NetworkScheduleResult:
        """
        Evaluate full network rates under a given pairing plan and slot assignment.
        custom_slot_mapping: optional list specifying slot ID for each pair in pairing_plan.pairs.
        """
        user_dict = {u.user_id: u for u in users}
        slot_records: List[SlotTransmissionRecord] = []
        user_rates: Dict[int, float] = {u.user_id: 0.0 for u in users}

        pairs = pairing_plan.pairs
        unpaired = pairing_plan.unpaired_users
        used_slots: set[int] = set()

        for idx, pair in enumerate(pairs):
            if idx >= self.num_slots:
                continue

            slot_id = custom_slot_mapping[idx] if custom_slot_mapping is not None else idx
            if not 0 <= slot_id < self.num_slots:
                slot_id = idx % self.num_slots
            if slot_id in used_slots:
                for candidate in range(self.num_slots):
                    if candidate not in used_slots:
                        slot_id = candidate
                        break
            used_slots.add(slot_id)

            u_a = user_dict[pair.user1_id]
            u_b = user_dict[pair.user2_id]
            selected_mode = self.select_communication_mode(u_a, u_b, slot_id)

            if selected_mode == "PD-NOMA":
                pd_res = self._compute_effective_pair_rates(
                    u_a,
                    u_b,
                    slot_id,
                    power_w=self.slot_power,
                    bandwidth_hz=self.slot_bandwidth,
                    noise_power_w=self.noise_power_per_slot,
                )
                user_rates[pd_res.weak_user_id] = pd_res.rate_weak_bps
                user_rates[pd_res.strong_user_id] = pd_res.rate_strong_bps
                record = SlotTransmissionRecord(
                    slot_id=slot_id,
                    mode="PD-NOMA",
                    assigned_user_ids=[pd_res.weak_user_id, pd_res.strong_user_id],
                    power_w=self.slot_power,
                    bandwidth_hz=self.slot_bandwidth,
                    slot_sum_rate_bps=pd_res.pair_sum_rate_bps,
                    per_user_rates_bps={
                        pd_res.weak_user_id: pd_res.rate_weak_bps,
                        pd_res.strong_user_id: pd_res.rate_strong_bps,
                    },
                    pd_pair_details=pd_res,
                )
            else:
                scma_rate = self._compute_scma_pair_rate(
                    u_a,
                    u_b,
                    slot_id,
                    power_w=self.slot_power,
                    bandwidth_hz=self.slot_bandwidth,
                    noise_power_w=self.noise_power_per_slot,
                )
                avg_rate = scma_rate / 2.0
                user_rates[u_a.user_id] = avg_rate
                user_rates[u_b.user_id] = avg_rate
                record = SlotTransmissionRecord(
                    slot_id=slot_id,
                    mode="SCMA",
                    assigned_user_ids=[u_a.user_id, u_b.user_id],
                    power_w=self.slot_power,
                    bandwidth_hz=self.slot_bandwidth,
                    slot_sum_rate_bps=scma_rate,
                    per_user_rates_bps={u_a.user_id: avg_rate, u_b.user_id: avg_rate},
                    pd_pair_details=None,
                )

            slot_records.append(record)

        # Remaining slots can host unpaired users (OMA mode)
        used_slots_for_oma = {rec.slot_id for rec in slot_records}
        for idx, u_id in enumerate(unpaired):
            if len(used_slots_for_oma) >= self.num_slots:
                break
            slot_id = 0
            while slot_id in used_slots_for_oma and slot_id < self.num_slots:
                slot_id += 1
            if slot_id >= self.num_slots:
                break
            used_slots_for_oma.add(slot_id)
            u = user_dict[u_id]
            oma_rate = self.pd_engine.compute_single_user_oma_rate(
                u,
                power_w=self.slot_power,
                bandwidth_hz=self.slot_bandwidth,
                noise_power_w=self.noise_power_per_slot * self._slot_noise_factor(slot_id),
            )
            user_rates[u_id] = oma_rate

            record = SlotTransmissionRecord(
                slot_id=slot_id,
                mode="OMA_SINGLE",
                assigned_user_ids=[u_id],
                power_w=self.slot_power,
                bandwidth_hz=self.slot_bandwidth,
                slot_sum_rate_bps=oma_rate,
                per_user_rates_bps={u_id: oma_rate},
                pd_pair_details=None,
            )
            slot_records.append(record)

        # Unallocated slots remain IDLE
        slot_ids_in_use = {rec.slot_id for rec in slot_records}
        total_active_slots = len(slot_records)
        for empty_slot_id in range(self.num_slots):
            if empty_slot_id not in slot_ids_in_use:
                slot_records.append(
                    SlotTransmissionRecord(
                        slot_id=empty_slot_id,
                        mode="IDLE",
                        assigned_user_ids=[],
                        power_w=0.0,
                        bandwidth_hz=self.slot_bandwidth,
                        slot_sum_rate_bps=0.0,
                        per_user_rates_bps={},
                    )
                )

        rates_list = list(user_rates.values())
        total_sum_rate = float(np.sum(rates_list))

        sum_r = float(np.sum(rates_list))
        sum_sq = float(np.sum(np.array(rates_list) ** 2))
        n = len(users)
        jains = (sum_r ** 2) / (n * sum_sq) if sum_sq > 0 else 1.0

        outages = 0
        for u in users:
            if user_rates[u.user_id] < u.min_rate_req_bps:
                outages += 1
        outage_prob = outages / n if n > 0 else 0.0

        return NetworkScheduleResult(
            slots=slot_records,
            total_sum_rate_bps=total_sum_rate,
            per_user_rates_bps=user_rates,
            jains_fairness=float(jains),
            active_slots=total_active_slots,
            total_slots=self.num_slots,
            slot_utilization=float(total_active_slots / self.num_slots),
            outage_user_count=outages,
            outage_probability=float(outage_prob),
        )
