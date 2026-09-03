"""Optional JAX/MJX helpers for compiling batched robot simulation steps.

This module intentionally has no JAX import at package import time.  The
existing MuJoCo Python wrapper continues to work without JAX; GPU MJX users
can install the ``mjx`` extra and pass pure reset/step functions here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def require_jax() -> tuple[Any, Any]:
    """Return ``(jax, jax.numpy)`` or explain how to enable batched rollouts."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "MJX support is optional. Install with `uv sync --extra mjx` on a "
            "JAX-supported platform, then construct the MJX model with "
            "mujoco.mjx.put_model`."
        ) from exc
    return jax, jnp


def require_mjx() -> Any:
    """Return MuJoCo's MJX module or explain how to install the extra."""
    try:
        from mujoco import mjx
    except ImportError as exc:
        raise ImportError(
            "MJX support is optional. Install with `uv sync --extra mjx`; "
            "the separate `mujoco-mjx` package provides `mujoco.mjx`."
        ) from exc
    return mjx


def make_batched_step(
    step_fn: Callable[[Any, Any], Any], batch_size: int
) -> Callable[[Any, Any], Any]:
    """JIT and vmap a pure MJX ``step_fn(data, action) -> next_data``.

    ``data`` and ``action`` must have the leading environment axis.  Keeping
    environment construction separate means task-specific resets, rewards,
    termination, and domain randomization stay transparent and testable.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    jax, _ = require_jax()
    return jax.jit(jax.vmap(step_fn, in_axes=(0, 0)))
