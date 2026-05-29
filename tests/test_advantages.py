"""Unit tests for the reward -> return -> advantage -> label math."""

import numpy as np
import pytest

from recap.advantages import (
    AdvantageConfig,
    compute_advantage_labels,
    discounted_returns,
    labels_from_advantages,
)


def test_discounted_returns_single_episode():
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    ep = np.array([0, 0, 0])
    out = discounted_returns(rewards, ep, gamma=0.5)
    # R_2 = 1; R_1 = 1 + .5*1 = 1.5; R_0 = 1 + .5*1.5 = 1.75
    np.testing.assert_allclose(out, [1.75, 1.5, 1.0], rtol=1e-6)


def test_discounted_returns_resets_across_episodes():
    rewards = np.array([1.0, 1.0, 5.0, 5.0], dtype=np.float32)
    ep = np.array([0, 0, 1, 1])
    out = discounted_returns(rewards, ep, gamma=1.0)
    # Episode 0: [2, 1]; episode 1: [10, 5]. No leakage across the boundary.
    np.testing.assert_allclose(out, [2.0, 1.0, 10.0, 5.0], rtol=1e-6)


def test_gamma_one_is_plain_sum():
    rewards = np.array([3.0, -1.0, 2.0], dtype=np.float32)
    ep = np.array([0, 0, 0])
    out = discounted_returns(rewards, ep, gamma=1.0)
    np.testing.assert_allclose(out, [4.0, 1.0, 2.0], rtol=1e-6)


def test_labels_hit_target_fraction():
    rng = np.random.default_rng(0)
    adv = rng.normal(size=10_000)
    labels, eps = labels_from_advantages(adv, positive_fraction=0.30)
    frac = labels.mean()
    assert abs(frac - 0.30) < 0.02, f"got {frac}"
    # epsilon should be ~ the 70th percentile of a standard normal (~0.524).
    assert 0.4 < eps < 0.65


def test_labels_are_binary_int8():
    adv = np.linspace(-1, 1, 100)
    labels, _ = labels_from_advantages(adv, positive_fraction=0.5)
    assert labels.dtype == np.int8
    assert set(np.unique(labels)).issubset({0, 1})


def test_compute_advantage_labels_end_to_end():
    rewards = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    values = np.zeros(6, dtype=np.float32)
    ep = np.array([0, 0, 0, 1, 1, 1])
    out = compute_advantage_labels(
        rewards, values, ep, AdvantageConfig(gamma=1.0, positive_fraction=0.5)
    )
    # With V=0, advantage == return. Each episode: [1,1,1] return-to-go (gamma=1).
    np.testing.assert_allclose(out["return"], [1, 1, 1, 1, 1, 1], rtol=1e-6)
    np.testing.assert_allclose(out["advantage"], [1, 1, 1, 1, 1, 1], rtol=1e-6)
    assert out["advantage_label"].shape == (6,)


def test_advantage_subtracts_value_baseline():
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([10.0, 10.0], dtype=np.float32)  # over-estimating baseline
    ep = np.array([0, 0])
    out = compute_advantage_labels(
        rewards, values, ep, AdvantageConfig(gamma=1.0, positive_fraction=0.5)
    )
    # returns are [2, 1]; advantages = returns - 10 = [-8, -9].
    np.testing.assert_allclose(out["advantage"], [-8.0, -9.0], rtol=1e-6)


def test_config_validation():
    with pytest.raises(ValueError):
        AdvantageConfig(gamma=0.0)
    with pytest.raises(ValueError):
        AdvantageConfig(gamma=1.5)
    with pytest.raises(ValueError):
        AdvantageConfig(positive_fraction=0.0)
    with pytest.raises(ValueError):
        AdvantageConfig(positive_fraction=1.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        discounted_returns(np.array([1.0, 2.0]), np.array([0]), gamma=0.9)
