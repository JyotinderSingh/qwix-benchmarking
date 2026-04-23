"""Throughput measurement (tokens / second)."""

import time
from typing import Any

import jax
import jax.numpy as jnp


def measure_throughput(
    forward_fn: Any,
    batch_size: int = 1,
    seq_len: int = 512,
    warmup_steps: int = 3,
    bench_steps: int = 10,
) -> dict[str, float]:
    """Measure forward-pass throughput.

    Args:
        forward_fn: Callable ``(tokens) -> logits``.
        batch_size: Number of sequences per batch.
        seq_len: Sequence length.
        warmup_steps: JIT warm-up iterations (excluded from timing).
        bench_steps: Measured iterations.

    Returns:
        Dict with ``"tokens_per_sec"``, ``"batches_per_sec"``,
        ``"avg_batch_time_ms"``, and ``"total_tokens"``.
    """
    tokens = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    total_tokens_per_step = batch_size * seq_len

    # Warm up (JIT compile + caches).
    for _ in range(warmup_steps):
        out = forward_fn(tokens)
        jax.block_until_ready(out)

    # Timed run.
    start = time.perf_counter()
    for _ in range(bench_steps):
        out = forward_fn(tokens)
        jax.block_until_ready(out)
    elapsed = time.perf_counter() - start

    total_tokens = total_tokens_per_step * bench_steps
    avg_batch_time = elapsed / bench_steps

    return {
        "tokens_per_sec": total_tokens / elapsed,
        "batches_per_sec": bench_steps / elapsed,
        "avg_batch_time_ms": avg_batch_time * 1000,
        "total_tokens": total_tokens,
    }
