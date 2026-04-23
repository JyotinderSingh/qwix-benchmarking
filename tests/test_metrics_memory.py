"""Tests for measure_memory."""

import jax.numpy as jnp

from qwix_bench.metrics.memory import measure_memory


def _identity(tokens):
    return tokens


def test_memory_returns_expected_keys():
    out = measure_memory(_identity, sample_input=jnp.zeros((1, 16), dtype=jnp.int32))
    expected = {
        "bytes_before",
        "peak_bytes",
        "peak_delta_bytes",
        "peak_delta_mb",
        "inference_bytes",
        "inference_mb",
    }
    assert set(out) == expected
    # Deltas must be non-negative; we don't assert absolute values on CPU
    # because memory_stats() may not be implemented.
    assert out["peak_delta_bytes"] >= 0
    assert out["peak_delta_mb"] >= 0
    assert out["inference_bytes"] >= 0
    assert out["inference_mb"] >= 0


def test_memory_default_sample_input():
    out = measure_memory(_identity)
    assert "peak_delta_bytes" in out
