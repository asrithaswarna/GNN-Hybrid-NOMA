"""
Evaluation and benchmarking module.
Evaluates OMA, Random NOMA, Near-Far NOMA, Greedy NOMA, Oracle, and GNN across
identical channel realizations for rigorous academic benchmarking.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
import time
import numpy as np
import pandas as pd
import torch

from src.config import ProjectConfig
from src.channel_model import WirelessNetwork, UserChannelState
from src.graph_builder import GraphBuilder
from src.noma_pd import PDNOMAEngine
from src.noma_scma import SCMAEngine, SCMAResult
from src.pairing import UserPairingManager, PairingPlan
from src.slot_assignment import SlotAssignmentEngine, NetworkScheduleResult
from src.optimization import OptimizationOracle
from src.gnn_model import HybridNOMAGNN


BENCHMARK_METHOD_ORDER = (
    "Orthogonal Multiple Access (OMA)",
    "Random Pairing PD-NOMA",
    "Near-Far Channel Gain NOMA",
    "Greedy Sum-Rate NOMA",
    "Optimization Oracle",
    "Proposed GNN-Based NOMA",
)


def _bpsk_symbol(bit: int) -> float:
    return 1.0 - 2.0 * bit


def _detect_bpsk(received: complex, channel: complex, amplitude: float) -> int:
    """Coherent maximum-likelihood BPSK detection."""
    return int(abs(received - channel * amplitude) > abs(received + channel * amplitude))


def _detect_weak_noma_bit(
    received: complex,
    channel: complex,
    weak_amplitude: float,
    strong_amplitude: float,
    noise_variance: float,
) -> int:
    """Marginal-ML detection of the weak BPSK bit with the strong bit unknown."""
    scores = []
    for weak_bit in (0, 1):
        distances = [
            abs(
                received
                - channel
                * (
                    weak_amplitude * _bpsk_symbol(weak_bit)
                    + strong_amplitude * _bpsk_symbol(strong_bit)
                )
            ) ** 2
            for strong_bit in (0, 1)
        ]
        scores.append(np.logaddexp(-distances[0] / noise_variance, -distances[1] / noise_variance))
    return int(scores[1] > scores[0])


@dataclass
class MethodBenchmarkMetrics:
    """Statistical summary of a pairing method over test trials."""
    method_name: str
    sum_rate_mean_mbps: float
    sum_rate_std_mbps: float
    user_rate_mean_mbps: float
    user_rate_std_mbps: float
    min_user_rate_mean_mbps: float
    jains_fairness_mean: float
    jains_fairness_std: float
    outage_probability_mean: float
    slot_utilization_mean: float
    inference_time_ms: float
    validity_rate: float


class NetworkEvaluator:
    """Runs comprehensive comparative evaluations across all baseline and proposed algorithms."""

    def __init__(self, cfg: ProjectConfig, gnn_model: Optional[HybridNOMAGNN] = None):
        self.cfg = cfg
        self.gnn_model = gnn_model
        self.network = WirelessNetwork(cfg.network, cfg.channel, seed=cfg.system.seed + 999)
        self.pd_engine = PDNOMAEngine(cfg.noma_pd)
        self.scma_engine = SCMAEngine(cfg.noma_scma)
        self.slot_engine = SlotAssignmentEngine(
            num_slots=cfg.network.num_slots,
            total_bandwidth_hz=cfg.network.total_bandwidth,
            total_power_w=cfg.network.total_transmit_power_w,
            noise_power_per_slot=self.network.noise_power_per_slot,
            pd_engine=self.pd_engine,
            gain_ratio_threshold=cfg.noma_pd.gain_ratio_threshold,
        )
        self.pairing_mgr = UserPairingManager(self.pd_engine)
        self.oracle = OptimizationOracle(
            slot_engine=self.slot_engine,
            pd_engine=self.pd_engine,
            fairness_weight=cfg.optimization.fairness_weight,
        )
        self.graph_builder = GraphBuilder(max_cell_radius=cfg.network.cell_radius_max)

    def evaluate_test_set(
        self,
        num_test_instances: int = 50,
        seed_offset: int = 2000,
        scenarios: Optional[List[List[UserChannelState]]] = None,
    ) -> Dict[str, MethodBenchmarkMetrics]:
        """
        Evaluate all methods on the exact same test channel realizations.
        """
        methods = [
            "Orthogonal Multiple Access (OMA)",
            "Random Pairing PD-NOMA",
            "Near-Far Channel Gain NOMA",
            "Greedy Sum-Rate NOMA",
            "Optimization Oracle",
        ]
        if self.gnn_model is not None:
            methods.append("Proposed GNN-Based NOMA")

        # Accumulator dictionaries
        sum_rates: Dict[str, List[float]] = {m: [] for m in methods}
        avg_rates: Dict[str, List[float]] = {m: [] for m in methods}
        min_rates: Dict[str, List[float]] = {m: [] for m in methods}
        fairness: Dict[str, List[float]] = {m: [] for m in methods}
        outages: Dict[str, List[float]] = {m: [] for m in methods}
        utilizations: Dict[str, List[float]] = {m: [] for m in methods}
        latencies: Dict[str, List[float]] = {m: [] for m in methods}
        validities: Dict[str, List[bool]] = {m: [] for m in methods}

        evaluation_count = len(scenarios) if scenarios is not None else num_test_instances
        for idx in range(evaluation_count):
            if scenarios is None:
                self.network.set_seed(seed_offset + idx)
                users = self.network.generate_users(
                    num_users=self.cfg.network.num_users,
                    min_rate_bps=self.cfg.optimization.min_rate_threshold_bps,
                )
            else:
                users = scenarios[idx]
            n = len(users)

            # 1. OMA Baseline (One user per slot up to S slots)
            t0 = time.perf_counter()
            oma_plan = PairingPlan(pairs=[], unpaired_users=[u.user_id for u in users], method_name="OMA")
            oma_sched = self.slot_engine.evaluate_schedule(oma_plan, users)
            lat_oma = (time.perf_counter() - t0) * 1000.0
            self._record_metric("Orthogonal Multiple Access (OMA)", oma_sched, lat_oma, oma_plan.validate(n),
                                sum_rates, avg_rates, min_rates, fairness, outages, utilizations, latencies, validities)

            # 2. Random Pairing
            t0 = time.perf_counter()
            rand_plan = self.pairing_mgr.random_pairing(users)
            rand_sched = self.slot_engine.evaluate_schedule(rand_plan, users)
            lat_rand = (time.perf_counter() - t0) * 1000.0
            self._record_metric("Random Pairing PD-NOMA", rand_sched, lat_rand, rand_plan.validate(n),
                                sum_rates, avg_rates, min_rates, fairness, outages, utilizations, latencies, validities)

            # 3. Near-Far Channel Gain
            t0 = time.perf_counter()
            nf_plan = self.pairing_mgr.channel_gain_pairing(users, strategy="near_far")
            nf_sched = self.slot_engine.evaluate_schedule(nf_plan, users)
            lat_nf = (time.perf_counter() - t0) * 1000.0
            self._record_metric("Near-Far Channel Gain NOMA", nf_sched, lat_nf, nf_plan.validate(n),
                                sum_rates, avg_rates, min_rates, fairness, outages, utilizations, latencies, validities)

            # 4. Greedy Sum-Rate
            t0 = time.perf_counter()
            greedy_plan = self.pairing_mgr.greedy_sum_rate_pairing(
                users,
                power_w=self.slot_engine.slot_power,
                bandwidth_hz=self.slot_engine.slot_bandwidth,
                noise_power_w=self.slot_engine.noise_power_per_slot,
            )
            greedy_sched = self.slot_engine.evaluate_schedule(greedy_plan, users)
            lat_greedy = (time.perf_counter() - t0) * 1000.0
            self._record_metric("Greedy Sum-Rate NOMA", greedy_sched, lat_greedy, greedy_plan.validate(n),
                                sum_rates, avg_rates, min_rates, fairness, outages, utilizations, latencies, validities)

            # 5. Optimization Oracle
            t0 = time.perf_counter()
            if n <= 10:
                oracle_plan, oracle_sched = self.oracle.solve_exhaustive(users)
            else:
                oracle_plan, oracle_sched = self.oracle.solve_max_weight_matching(users)
            lat_oracle = (time.perf_counter() - t0) * 1000.0
            self._record_metric("Optimization Oracle", oracle_sched, lat_oracle, oracle_plan.validate(n),
                                sum_rates, avg_rates, min_rates, fairness, outages, utilizations, latencies, validities)

            # 6. Proposed GNN
            if self.gnn_model is not None:
                t0 = time.perf_counter()
                graph = self.graph_builder.build_graph(users)
                self.gnn_model.eval()
                with torch.no_grad():
                    aff_probs = self.gnn_model.predict_pairing_probabilities(
                        graph.node_features, graph.edge_index, graph.edge_features
                    ).cpu().numpy()
                    slot_logits = self.gnn_model.predict_slot_logits(
                        graph.node_features, graph.edge_index, graph.edge_features
                    ).cpu()
                gnn_plan = self.pairing_mgr.gnn_affinity_pairing(aff_probs, users)
                gnn_slot_mapping = self.slot_engine.resolve_gnn_slot_mapping(gnn_plan, users, slot_logits)
                gnn_sched = self.slot_engine.evaluate_schedule(gnn_plan, users, custom_slot_mapping=gnn_slot_mapping)
                lat_gnn = (time.perf_counter() - t0) * 1000.0
                self._record_metric("Proposed GNN-Based NOMA", gnn_sched, lat_gnn, gnn_plan.validate(n),
                                    sum_rates, avg_rates, min_rates, fairness, outages, utilizations, latencies, validities)

        # Aggregate statistics
        results: Dict[str, MethodBenchmarkMetrics] = {}
        for m in methods:
            results[m] = MethodBenchmarkMetrics(
                method_name=m,
                sum_rate_mean_mbps=float(np.mean(sum_rates[m])) / 1.0e6,
                sum_rate_std_mbps=float(np.std(sum_rates[m])) / 1.0e6,
                user_rate_mean_mbps=float(np.mean(avg_rates[m])) / 1.0e6,
                user_rate_std_mbps=float(np.std(avg_rates[m])) / 1.0e6,
                min_user_rate_mean_mbps=float(np.mean(min_rates[m])) / 1.0e6,
                jains_fairness_mean=float(np.mean(fairness[m])),
                jains_fairness_std=float(np.std(fairness[m])),
                outage_probability_mean=float(np.mean(outages[m])),
                slot_utilization_mean=float(np.mean(utilizations[m])) * 100.0,
                inference_time_ms=float(np.mean(latencies[m])),
                validity_rate=float(np.mean(validities[m])) * 100.0,
            )

        return results

    def _record_metric(
        self,
        method: str,
        sched: NetworkScheduleResult,
        latency_ms: float,
        is_valid: bool,
        sum_rates: Dict[str, List[float]],
        avg_rates: Dict[str, List[float]],
        min_rates: Dict[str, List[float]],
        fairness: Dict[str, List[float]],
        outages: Dict[str, List[float]],
        utilizations: Dict[str, List[float]],
        latencies: Dict[str, List[float]],
        validities: Dict[str, List[bool]],
    ) -> None:
        user_rates_arr = list(sched.per_user_rates_bps.values())
        sum_rates[method].append(sched.total_sum_rate_bps)
        avg_rates[method].append(float(np.mean(user_rates_arr)) if user_rates_arr else 0.0)
        min_rates[method].append(float(np.min(user_rates_arr)) if user_rates_arr else 0.0)
        fairness[method].append(sched.jains_fairness)
        outages[method].append(sched.outage_probability)
        utilizations[method].append(sched.slot_utilization)
        latencies[method].append(latency_ms)
        validities[method].append(is_valid)

    def evaluate_scma_snr_sweep(
        self,
        snr_db_range: List[float] = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
        num_blocks: int = 100,
    ) -> List[SCMAResult]:
        """Evaluate SCMA MPA detector across SNR levels."""
        dummy_gains = np.ones(self.scma_engine.J)
        scma_results: List[SCMAResult] = []
        for snr in snr_db_range:
            res = self.scma_engine.simulate_transmission(
                channel_gains=dummy_gains,
                snr_db=snr,
                num_blocks=num_blocks,
                bandwidth_hz=self.cfg.network.total_bandwidth,
            )
            scma_results.append(res)
        return scma_results

    def evaluate_ber_snr_sweep(
        self,
        snr_db_range: Optional[List[float]] = None,
        num_blocks: int = 100,
        num_test_instances: int = 50,
        seed_offset: int = 1000,
        scenarios: Optional[List[List[UserChannelState]]] = None,
    ) -> pd.DataFrame:
        """
        Compare uncoded BPSK BER for OMA and the pairing methods over common channels.

        Paired methods use two-user PD-NOMA superposition, marginal-ML weak-user
        detection, and hard SIC at the strong user. OMA uses orthogonal,
        half-slot transmissions. The SNR sweep normalizes AWGN to the mean
        received channel power in each scenario; the existing SCMA simulation
        and its QPSK/MPA assumptions are unchanged.
        """
        snrs = snr_db_range if snr_db_range is not None else [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if not snrs or any(not np.isfinite(snr) for snr in snrs):
            raise ValueError("snr_db_range must contain finite SNR values")

        if scenarios is None:
            network = WirelessNetwork(self.cfg.network, self.cfg.channel, seed=seed_offset)
            scenario_list: List[List[UserChannelState]] = []
            for idx in range(num_test_instances):
                network.set_seed(seed_offset + idx)
                scenario_list.append(network.generate_users(
                    num_users=self.cfg.network.num_users,
                    min_rate_bps=self.cfg.optimization.min_rate_threshold_bps,
                ))
        else:
            scenario_list = scenarios
        if not scenario_list:
            raise ValueError("BER evaluation requires at least one test scenario")

        method_names = list(BENCHMARK_METHOD_ORDER[:-1])
        if self.gnn_model is not None:
            method_names.append(BENCHMARK_METHOD_ORDER[-1])
        totals = {
            (float(snr), method): {"bit_errors": 0, "total_bits": 0}
            for snr in snrs
            for method in method_names
        }
        alpha_weak = self.cfg.noma_pd.alpha_weak
        alpha_strong = self.cfg.noma_pd.alpha_strong
        power = self.slot_engine.slot_power
        mean_gain_floor = 1e-30

        for scenario_idx, users in enumerate(scenario_list):
            if not users:
                raise ValueError(f"BER scenario {scenario_idx} contains no users")
            user_by_id = {user.user_id: user for user in users}
            if len(user_by_id) != len(users):
                raise ValueError(f"BER scenario {scenario_idx} contains duplicate user IDs")

            plans: Dict[str, PairingPlan] = {
                BENCHMARK_METHOD_ORDER[0]: PairingPlan(
                    pairs=[],
                    unpaired_users=[user.user_id for user in users],
                    method_name="OMA",
                )
            }
            plans[BENCHMARK_METHOD_ORDER[1]] = self.pairing_mgr.random_pairing(
                users,
                rng=np.random.default_rng(self.cfg.system.seed + seed_offset + scenario_idx),
            )
            plans[BENCHMARK_METHOD_ORDER[2]] = self.pairing_mgr.channel_gain_pairing(
                users, strategy="near_far"
            )
            plans[BENCHMARK_METHOD_ORDER[3]] = self.pairing_mgr.greedy_sum_rate_pairing(
                users,
                power_w=self.slot_engine.slot_power,
                bandwidth_hz=self.slot_engine.slot_bandwidth,
                noise_power_w=self.slot_engine.noise_power_per_slot,
            )
            if len(users) <= 10:
                oracle_plan, _ = self.oracle.solve_exhaustive(users)
            else:
                oracle_plan, _ = self.oracle.solve_max_weight_matching(users)
            plans[BENCHMARK_METHOD_ORDER[4]] = oracle_plan

            if self.gnn_model is not None:
                graph = self.graph_builder.build_graph(users)
                model_device = next(self.gnn_model.parameters()).device
                self.gnn_model.eval()
                with torch.no_grad():
                    affinity = self.gnn_model.predict_pairing_probabilities(
                        graph.node_features.to(model_device),
                        graph.edge_index.to(model_device),
                        graph.edge_features.to(model_device),
                    ).cpu().numpy()
                    slot_logits = self.gnn_model.predict_slot_logits(
                        graph.node_features.to(model_device),
                        graph.edge_index.to(model_device),
                        graph.edge_features.to(model_device),
                    ).cpu()
                gnn_plan = self.pairing_mgr.gnn_affinity_pairing(affinity, users)
                gnn_slot_mapping = self.slot_engine.resolve_gnn_slot_mapping(
                    gnn_plan, users, slot_logits
                )
                plans[BENCHMARK_METHOD_ORDER[5]] = gnn_plan
            else:
                gnn_slot_mapping = []

            mean_channel_gain = max(
                float(np.mean([user.channel_gain for user in users])),
                mean_gain_floor,
            )
            for snr_idx, snr_db in enumerate(snrs):
                noise_variance = power * mean_channel_gain / (10.0 ** (float(snr_db) / 10.0))
                noise_variance = max(noise_variance, 1e-30)
                rng = np.random.default_rng(
                    self.cfg.system.seed
                    + seed_offset * 100_000
                    + scenario_idx * 10_000
                    + snr_idx
                )
                for _ in range(num_blocks):
                    tx_bits = rng.integers(0, 2, size=len(users))
                    standard_noise = (
                        rng.normal(size=len(users)) + 1j * rng.normal(size=len(users))
                    ) / np.sqrt(2.0)

                    for method in method_names:
                        plan = plans[method]
                        if not plan.validate(len(users)):
                            raise ValueError(f"Invalid pairing plan for BER method '{method}'")
                        errors = 0
                        detected_users = set()

                        if method == BENCHMARK_METHOD_ORDER[0]:
                            # OMA shares each slot orthogonally in time; all users
                            # therefore contribute the same number of transmitted bits.
                            for user_idx, user in enumerate(users):
                                slot_id = user_idx % self.slot_engine.num_slots
                                slot_gain = self.slot_engine._slot_quality_factor(slot_id)
                                slot_noise = self.slot_engine._slot_noise_factor(slot_id)
                                channel = user.channel_complex * np.sqrt(slot_gain)
                                slot_noise_variance = noise_variance * slot_noise
                                rx = (
                                    channel * np.sqrt(power) * _bpsk_symbol(int(tx_bits[user_idx]))
                                    + standard_noise[user_idx] * np.sqrt(slot_noise_variance)
                                )
                                detected = _detect_bpsk(rx, channel, np.sqrt(power))
                                errors += int(detected != tx_bits[user_idx])
                                detected_users.add(user.user_id)
                        else:
                            for pair_idx, pair in enumerate(plan.pairs):
                                slot_id = (
                                    gnn_slot_mapping[pair_idx]
                                    if method == BENCHMARK_METHOD_ORDER[5]
                                    else pair_idx % self.slot_engine.num_slots
                                )
                                slot_gain = self.slot_engine._slot_quality_factor(slot_id)
                                slot_noise = self.slot_engine._slot_noise_factor(slot_id)
                                slot_noise_variance = max(noise_variance * slot_noise, 1e-30)
                                first = user_by_id[pair.user1_id]
                                second = user_by_id[pair.user2_id]
                                weak, strong = (
                                    (first, second)
                                    if first.channel_gain <= second.channel_gain
                                    else (second, first)
                                )
                                weak_idx = next(i for i, user in enumerate(users) if user.user_id == weak.user_id)
                                strong_idx = next(i for i, user in enumerate(users) if user.user_id == strong.user_id)
                                weak_channel = weak.channel_complex * np.sqrt(slot_gain)
                                strong_channel = strong.channel_complex * np.sqrt(slot_gain)
                                weak_amplitude = np.sqrt(power * alpha_weak)
                                strong_amplitude = np.sqrt(power * alpha_strong)
                                weak_rx = (
                                    weak_channel
                                    * (
                                        weak_amplitude * _bpsk_symbol(int(tx_bits[weak_idx]))
                                        + strong_amplitude * _bpsk_symbol(int(tx_bits[strong_idx]))
                                    )
                                    + standard_noise[weak_idx] * np.sqrt(slot_noise_variance)
                                )
                                strong_rx = (
                                    strong_channel
                                    * (
                                        weak_amplitude * _bpsk_symbol(int(tx_bits[weak_idx]))
                                        + strong_amplitude * _bpsk_symbol(int(tx_bits[strong_idx]))
                                    )
                                    + standard_noise[strong_idx] * np.sqrt(slot_noise_variance)
                                )
                                weak_detected = _detect_weak_noma_bit(
                                    weak_rx,
                                    weak_channel,
                                    weak_amplitude,
                                    strong_amplitude,
                                    slot_noise_variance,
                                )
                                strong_weak_detected = _detect_weak_noma_bit(
                                    strong_rx,
                                    strong_channel,
                                    weak_amplitude,
                                    strong_amplitude,
                                    slot_noise_variance,
                                )
                                cancellation_amplitude = (
                                    1.0 - np.sqrt(self.cfg.noma_pd.sic_residual_factor)
                                ) * weak_amplitude
                                strong_after_sic = strong_rx - (
                                    strong_channel
                                    * cancellation_amplitude
                                    * _bpsk_symbol(strong_weak_detected)
                                )
                                strong_detected = _detect_bpsk(
                                    strong_after_sic,
                                    strong_channel,
                                    strong_amplitude,
                                )
                                errors += int(weak_detected != tx_bits[weak_idx])
                                errors += int(strong_detected != tx_bits[strong_idx])
                                detected_users.add(weak.user_id)
                                detected_users.add(strong.user_id)

                            # Pairings with an odd user count send the remaining
                            # user orthogonally, as the benchmark scheduler does.
                            for unpaired_idx, user_id in enumerate(plan.unpaired_users):
                                if user_id in detected_users:
                                    continue
                                user_idx = next(i for i, user in enumerate(users) if user.user_id == user_id)
                                slot_id = (len(plan.pairs) + unpaired_idx) % self.slot_engine.num_slots
                                slot_gain = self.slot_engine._slot_quality_factor(slot_id)
                                slot_noise_variance = max(
                                    noise_variance * self.slot_engine._slot_noise_factor(slot_id),
                                    1e-30,
                                )
                                channel = user_by_id[user_id].channel_complex * np.sqrt(slot_gain)
                                rx = (
                                    channel * np.sqrt(power) * _bpsk_symbol(int(tx_bits[user_idx]))
                                    + standard_noise[user_idx] * np.sqrt(slot_noise_variance)
                                )
                                detected = _detect_bpsk(rx, channel, np.sqrt(power))
                                errors += int(detected != tx_bits[user_idx])
                                detected_users.add(user_id)

                        if detected_users != set(user_by_id):
                            raise RuntimeError(
                                f"BER method '{method}' did not transmit for every user "
                                f"in scenario {scenario_idx}"
                            )
                        totals[(float(snr_db), method)]["bit_errors"] += errors
                        totals[(float(snr_db), method)]["total_bits"] += len(users)

        rows = []
        for snr_db in snrs:
            for method in method_names:
                counts = totals[(float(snr_db), method)]
                if counts["total_bits"] == 0:
                    raise RuntimeError(f"No bits were transmitted for BER method '{method}'")
                rows.append({
                    "snr_db": float(snr_db),
                    "method": method,
                    "bit_errors": counts["bit_errors"],
                    "total_bits": counts["total_bits"],
                    "ber": counts["bit_errors"] / counts["total_bits"],
                })
        return pd.DataFrame(rows)

    def to_dataframe(self, benchmark_results: Dict[str, MethodBenchmarkMetrics]) -> pd.DataFrame:
        """Convert benchmark metrics to structured DataFrame."""
        rows = [asdict(m) for m in benchmark_results.values()]
        df = pd.DataFrame(rows)
        return df
