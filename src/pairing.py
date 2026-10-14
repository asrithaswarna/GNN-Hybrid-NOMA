"""
User pairing algorithms for Hybrid NOMA networks.
Implements Random, Near-Far Channel-Gain, Greedy Sum-Rate, and GNN-driven pairing.
Strictly guarantees no duplicate assignments and robust odd-user handling.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import numpy as np
from src.channel_model import UserChannelState
from src.noma_pd import PDNOMAEngine


@dataclass
class UserPair:
    """A pair of paired users."""
    user1_id: int
    user2_id: int
    predicted_affinity: float = 1.0


@dataclass
class PairingPlan:
    """The complete user pairing result for a network instance."""
    pairs: List[UserPair]
    unpaired_users: List[int]
    method_name: str

    @property
    def total_paired_users(self) -> int:
        return len(self.pairs) * 2

    def validate(self, total_users: int) -> bool:
        """Verify that no user is paired twice and all user IDs are within [0, total_users - 1]."""
        seen: Set[int] = set()
        for p in self.pairs:
            if p.user1_id == p.user2_id:
                return False
            if p.user1_id in seen or p.user2_id in seen:
                return False
            if not (0 <= p.user1_id < total_users) or not (0 <= p.user2_id < total_users):
                return False
            seen.add(p.user1_id)
            seen.add(p.user2_id)

        for u in self.unpaired_users:
            if u in seen or not (0 <= u < total_users):
                return False
            seen.add(u)

        return len(seen) == total_users


class UserPairingManager:
    """
    Implements multiple pairing strategies for NOMA networks.
    """

    def __init__(self, pd_engine: Optional[PDNOMAEngine] = None):
        self.pd_engine = pd_engine

    def random_pairing(
        self,
        users: List[UserChannelState],
        rng: Optional[np.random.Generator] = None,
    ) -> PairingPlan:
        """
        Baseline 1: Random Pairing.
        Permutes users randomly and groups adjacent pairs.
        """
        if rng is None:
            rng = np.random.default_rng()

        user_ids = [u.user_id for u in users]
        shuffled = rng.permutation(user_ids).tolist()

        pairs: List[UserPair] = []
        unpaired: List[int] = []

        n = len(shuffled)
        num_pairs = n // 2

        for i in range(num_pairs):
            u1 = shuffled[2 * i]
            u2 = shuffled[2 * i + 1]
            pairs.append(UserPair(user1_id=u1, user2_id=u2, predicted_affinity=1.0))

        if n % 2 != 0:
            unpaired.append(shuffled[-1])

        plan = PairingPlan(pairs=pairs, unpaired_users=unpaired, method_name="Random Pairing")
        assert plan.validate(len(users)), "Random pairing validation failed"
        return plan

    def channel_gain_pairing(
        self,
        users: List[UserChannelState],
        strategy: str = "near_far",
    ) -> PairingPlan:
        """
        Baseline 2: Channel-Gain-Based Pairing.
        'near_far': Standard conventional NOMA baseline (pairs highest with lowest gain).
        'adjacent': Pairs adjacent users in gain-sorted order.
        """
        sorted_users = sorted(users, key=lambda u: u.channel_gain)
        sorted_ids = [u.user_id for u in sorted_users]
        n = len(sorted_ids)

        pairs: List[UserPair] = []
        unpaired: List[int] = []

        if strategy == "near_far":
            # Pair user i with user (n - 1 - i)
            num_pairs = n // 2
            for i in range(num_pairs):
                u_weak = sorted_ids[i]
                u_strong = sorted_ids[n - 1 - i]
                pairs.append(UserPair(user1_id=u_weak, user2_id=u_strong, predicted_affinity=1.0))
            if n % 2 != 0:
                # Middle user remains unpaired
                unpaired.append(sorted_ids[num_pairs])
        else:
            # Adjacent pairing
            num_pairs = n // 2
            for i in range(num_pairs):
                pairs.append(
                    UserPair(
                        user1_id=sorted_ids[2 * i],
                        user2_id=sorted_ids[2 * i + 1],
                        predicted_affinity=1.0,
                    )
                )
            if n % 2 != 0:
                unpaired.append(sorted_ids[-1])

        plan = PairingPlan(pairs=pairs, unpaired_users=unpaired, method_name=f"Channel Gain ({strategy})")
        assert plan.validate(n), "Channel gain pairing validation failed"
        return plan

    def greedy_sum_rate_pairing(
        self,
        users: List[UserChannelState],
        power_w: float,
        bandwidth_hz: float,
        noise_power_w: float,
    ) -> PairingPlan:
        """
        Baseline 3: Greedy Maximum Sum-Rate Pairing.
        Iteratively picks candidate pair yielding the highest 2-user sum rate.
        """
        if self.pd_engine is None:
            raise ValueError("PDNOMAEngine required for greedy_sum_rate_pairing")

        n = len(users)
        user_dict = {u.user_id: u for u in users}
        active_ids = set(user_dict.keys())

        # Precompute candidate pair rates
        candidate_scores: List[Tuple[float, int, int]] = []
        u_list = list(user_dict.values())
        for i in range(n):
            for j in range(i + 1, n):
                res = self.pd_engine.compute_pair_rates(
                    u_list[i], u_list[j], power_w, bandwidth_hz, noise_power_w
                )
                candidate_scores.append((res.pair_sum_rate_bps, u_list[i].user_id, u_list[j].user_id))

        # Sort descending by sum rate
        candidate_scores.sort(key=lambda item: item[0], reverse=True)

        pairs: List[UserPair] = []
        for rate, u1, u2 in candidate_scores:
            if u1 in active_ids and u2 in active_ids:
                pairs.append(UserPair(user1_id=u1, user2_id=u2, predicted_affinity=rate))
                active_ids.remove(u1)
                active_ids.remove(u2)

        unpaired = sorted(list(active_ids))
        plan = PairingPlan(pairs=pairs, unpaired_users=unpaired, method_name="Greedy Sum-Rate")
        assert plan.validate(n), "Greedy pairing validation failed"
        return plan

    def gnn_affinity_pairing(
        self,
        affinity_matrix: np.ndarray,
        users: List[UserChannelState],
    ) -> PairingPlan:
        """
        Proposed Method: Converts continuous GNN-predicted affinity matrix into discrete pairs.
        Uses greedy maximum weight matching on affinity_matrix A[i, j].
        """
        n = len(users)
        active_users = set(range(n))

        # Symmetrize and remove diagonal
        sym_aff = 0.5 * (affinity_matrix + affinity_matrix.T)
        np.fill_diagonal(sym_aff, -np.inf)

        edges: List[Tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((float(sym_aff[i, j]), i, j))

        edges.sort(key=lambda x: x[0], reverse=True)

        pairs: List[UserPair] = []
        for weight, i, j in edges:
            if i in active_users and j in active_users:
                pairs.append(UserPair(user1_id=i, user2_id=j, predicted_affinity=weight))
                active_users.remove(i)
                active_users.remove(j)

        unpaired = sorted(list(active_users))
        plan = PairingPlan(pairs=pairs, unpaired_users=unpaired, method_name="GNN-Based Pairing")
        assert plan.validate(n), "GNN pairing validation failed"
        return plan
