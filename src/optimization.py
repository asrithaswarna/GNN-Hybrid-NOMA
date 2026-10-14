"""
Combinatorial optimization oracle and supervisory label generator.
Provides globally optimal user pairing via exhaustive search for small N (<= 10),
and Edmonds' maximum-weight matching for larger N.
"""

from typing import List, Tuple, Dict, Optional, Set
import itertools
import numpy as np
import networkx as nx
from src.channel_model import UserChannelState
from src.noma_pd import PDNOMAEngine
from src.pairing import PairingPlan, UserPair
from src.slot_assignment import SlotAssignmentEngine, NetworkScheduleResult


def generate_all_pair_partitions(elements: List[int]) -> List[List[Tuple[int, int]]]:
    """
    Recursively generates all distinct complete pairings of an even number of elements.
    For N=10, yields exactly 945 distinct pairing partitions.
    """
    if len(elements) < 2:
        return [[]]
    first = elements[0]
    rest = elements[1:]
    partitions = []
    for i, second in enumerate(rest):
        pair = (first, second)
        remaining = rest[:i] + rest[i + 1:]
        for sub_partition in generate_all_pair_partitions(remaining):
            partitions.append([pair] + sub_partition)
    return partitions


class OptimizationOracle:
    """
    Finds globally optimal or heuristic-optimal user pairings to maximize utility:
    Utility = Total_Sum_Rate + lambda_fair * Jain_Fairness
    """

    def __init__(
        self,
        slot_engine: SlotAssignmentEngine,
        pd_engine: PDNOMAEngine,
        fairness_weight: float = 0.2,
    ):
        self.slot_engine = slot_engine
        self.pd_engine = pd_engine
        self.fairness_weight = fairness_weight

    def solve_exhaustive(self, users: List[UserChannelState]) -> Tuple[PairingPlan, NetworkScheduleResult]:
        """
        Global Optimum Oracle: Exhaustive search over all pairing partitions.
        Handles both even and odd N.
        """
        n = len(users)
        user_ids = [u.user_id for u in users]

        best_utility = -1e9
        best_plan: Optional[PairingPlan] = None
        best_schedule: Optional[NetworkScheduleResult] = None

        if n % 2 == 0:
            candidate_partitions = [
                (partition, []) for partition in generate_all_pair_partitions(user_ids)
            ]
        else:
            # For odd N, select which user remains unpaired
            candidate_partitions = []
            for idx in range(n):
                unpaired_user = user_ids[idx]
                paired_pool = user_ids[:idx] + user_ids[idx + 1:]
                for partition in generate_all_pair_partitions(paired_pool):
                    candidate_partitions.append((partition, [unpaired_user]))

        for partition, unpaired in candidate_partitions:
            plan_pairs = [UserPair(user1_id=p[0], user2_id=p[1], predicted_affinity=1.0) for p in partition]
            candidate_plan = PairingPlan(
                pairs=plan_pairs,
                unpaired_users=unpaired,
                method_name="Optimization Oracle (Exhaustive)",
            )
            sched = self.slot_engine.evaluate_schedule(candidate_plan, users)
            # Utility function
            utility = sched.total_sum_rate_bps + self.fairness_weight * sched.jains_fairness * 1e6
            if utility > best_utility:
                best_utility = utility
                best_plan = candidate_plan
                best_schedule = sched

        assert best_plan is not None and best_schedule is not None
        return best_plan, best_schedule

    def solve_max_weight_matching(self, users: List[UserChannelState]) -> Tuple[PairingPlan, NetworkScheduleResult]:
        """
        Scalable Oracle: Edmonds' Blossom maximum-weight matching algorithm.
        Constructs a complete weighted graph where edge weights equal achievable 2-user sum rate.
        Complexity: O(V^3), scales efficiently to large N.
        """
        n = len(users)
        u_dict = {u.user_id: u for u in users}

        G = nx.Graph()
        G.add_nodes_from(u_dict.keys())

        u_list = list(u_dict.values())
        for i in range(n):
            for j in range(i + 1, n):
                res = self.pd_engine.compute_pair_rates(
                    u_list[i],
                    u_list[j],
                    power_w=self.slot_engine.slot_power,
                    bandwidth_hz=self.slot_engine.slot_bandwidth,
                    noise_power_w=self.slot_engine.noise_power_per_slot,
                )
                weight = res.pair_sum_rate_bps
                G.add_edge(u_list[i].user_id, u_list[j].user_id, weight=weight)

        # Maximum weight matching
        matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True, weight="weight")

        pairs: List[UserPair] = []
        matched_nodes: Set[int] = set()
        for u1, u2 in matching:
            pairs.append(UserPair(user1_id=u1, user2_id=u2, predicted_affinity=1.0))
            matched_nodes.add(u1)
            matched_nodes.add(u2)

        unpaired = sorted([uid for uid in u_dict.keys() if uid not in matched_nodes])
        plan = PairingPlan(pairs=pairs, unpaired_users=unpaired, method_name="Optimization Oracle (Blossom)")
        sched = self.slot_engine.evaluate_schedule(plan, users)
        return plan, sched

    def generate_labels(
        self,
        users: List[UserChannelState],
        method: str = "exhaustive",
    ) -> Tuple[np.ndarray, np.ndarray, PairingPlan]:
        """
        Generates ground-truth pair label matrix [N, N] and slot label vector [N]
        for GNN supervised training.
        """
        n = len(users)
        if method == "exhaustive" and n <= 10:
            plan, sched = self.solve_exhaustive(users)
        else:
            plan, sched = self.solve_max_weight_matching(users)

        # Pair target adjacency matrix [N, N]
        pair_matrix = np.zeros((n, n), dtype=np.float32)
        for p in plan.pairs:
            pair_matrix[p.user1_id, p.user2_id] = 1.0
            pair_matrix[p.user2_id, p.user1_id] = 1.0

        # Slot target vector [N]
        slot_vector = np.zeros(n, dtype=np.int64)
        for rec in sched.slots:
            for uid in rec.assigned_user_ids:
                slot_vector[uid] = rec.slot_id

        return pair_matrix, slot_vector, plan
