"""RECAP training-loop components for SmolVLA.

This package implements the policy-training side of RECAP
(RL with Experience and Corrections via Advantage-conditioned Policies,
arXiv:2511.14759) on top of LeRobot's SmolVLA.

Pipeline:
    classifier(frame) -> reward r_t
        -> discounted return R_t = sum_k gamma^k r_{t+k}
        -> value function V(s_t) regresses R_t
        -> advantage A_t = R_t - V(s_t)
        -> binary label I_t = 1[A_t > epsilon]   (epsilon set so ~30% positive)
        -> SmolVLA conditioned on "Advantage: positive/negative" token
           in the language prompt, with classifier-free-guidance dropout.

The modules here are intentionally framework-light so they can be unit-tested
without GPUs or the full LeRobot stack.
"""

from recap.advantages import (
    AdvantageConfig,
    compute_advantage_labels,
    discounted_returns,
)
from recap.classifier import RewardClassifier, StubRewardClassifier
from recap.value import ValueConfig, ValueMLP

__all__ = [
    "RewardClassifier",
    "StubRewardClassifier",
    "discounted_returns",
    "compute_advantage_labels",
    "AdvantageConfig",
    "ValueMLP",
    "ValueConfig",
]
