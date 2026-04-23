"""Tests for measure_latency and measure_throughput."""

from qwix_bench.metrics.latency import measure_latency
from qwix_bench.metrics.throughput import measure_throughput


def _identity(tokens):
    return tokens


def test_throughput_returns_expected_keys():
    out = measure_throughput(_identity, batch_size=2, seq_len=16, warmup_steps=1, bench_steps=3)
    assert set(out) == {
        "tokens_per_sec",
        "batches_per_sec",
        "avg_batch_time_ms",
        "total_tokens",
    }
    assert out["total_tokens"] == 2 * 16 * 3
    assert out["tokens_per_sec"] > 0
    assert out["batches_per_sec"] > 0
    assert out["avg_batch_time_ms"] > 0


def test_latency_returns_expected_keys_and_ordering():
    out = measure_latency(_identity, batch_size=1, seq_len=8, warmup_steps=1, bench_steps=5)
    assert set(out) == {
        "mean_ms",
        "median_ms",
        "p90_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
    }
    # Percentile / min / max ordering invariants.
    assert out["min_ms"] <= out["median_ms"] <= out["max_ms"]
    assert out["median_ms"] <= out["p90_ms"] <= out["p99_ms"]
    assert out["p99_ms"] <= out["max_ms"]
    assert out["mean_ms"] > 0
