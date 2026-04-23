"""Latency measurement (per-batch forward pass timing)."""

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def measure_latency(
    forward_fn: Any,
    batch_size: int = 1,
    seq_len: int = 512,
    warmup_steps: int = 3,
    bench_steps: int = 10,
) -> dict[str, float]:
    """Measure per-batch latency statistics.

    Args:
        forward_fn: Callable ``(tokens) -> logits``.
        batch_size: Number of sequences per batch.
        seq_len: Sequence length.
        warmup_steps: JIT warm-up iterations.
        bench_steps: Measured iterations.

    Returns:
        Dict with ``"mean_ms"``, ``"median_ms"``, ``"p90_ms"``,
        ``"p99_ms"``, ``"min_ms"``, ``"max_ms"``.
    """
    tokens = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    # Warm up.
    for _ in range(warmup_steps):
        out = forward_fn(tokens)
        jax.block_until_ready(out)

    # Collect per-step timings.
    times_ms: list[float] = []
    for _ in range(bench_steps):
        start = time.perf_counter()
        out = forward_fn(tokens)
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - start
        times_ms.append(elapsed * 1000)

    arr = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }
