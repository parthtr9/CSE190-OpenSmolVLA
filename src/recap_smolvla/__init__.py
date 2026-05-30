"""
recap_smolvla — RECAP advantage-conditioned offline RL for SmolVLA.

Modules
-------
rewards         sparse and dense reward functions (Equation 5 from π*0.6 paper)
value_function  MLP value function + return / training utilities
advantage       advantage computation and binary labeling
rollout         trajectory collection and MockEnv fallback
training        RECAP iteration loop and SmolVLA fine-tuning
scoring         VLA self-supervised trajectory scoring (Direction 2)
curriculum      difficulty-scheduled environment wrapper (Direction 3)
data            LeRobot dataset adapter for offline evaluation
"""

from recap_smolvla.rewards import sparse_reward_fn, dense_reward_fn
from recap_smolvla.value_function import ValueFunction, compute_returns, train_value_function
from recap_smolvla.advantage import compute_advantages, label_trajectories
from recap_smolvla.rollout import collect_rollout
from recap_smolvla.training import recap_training_iteration, finetune_smolvla
from recap_smolvla.data import (
    dataset_to_rollouts,
    split_rollouts,
    offline_eval_metrics,
    MockLeRobotDataset,
)

__all__ = [
    "sparse_reward_fn",
    "dense_reward_fn",
    "ValueFunction",
    "compute_returns",
    "train_value_function",
    "compute_advantages",
    "label_trajectories",
    "collect_rollout",
    "recap_training_iteration",
    "finetune_smolvla",
    "dataset_to_rollouts",
    "split_rollouts",
    "offline_eval_metrics",
    "MockLeRobotDataset",
]
