"""Tests for report.py — JSON, Markdown, CSV, baseline-delta math."""

import csv
import json

import jax.numpy as jnp
import numpy as np

from qwix_bench.report import (
    collect_metadata,
    format_report,
    format_report_markdown,
    save_csv,
    save_json,
)


def _sample_results():
    return {
        "baseline": {
            "model_size": {
                "total_bytes": 1000,
                "total_mb": 1.0,
                "param_count": 250,
                "dtype_bytes": {"float32": 1000},
            },
            "accuracy": {"loss": 1.0, "perplexity": 2.71828},
            "throughput": {"tokens_per_sec": 100.0, "avg_batch_time_ms": 10.0},
            "latency": {
                "mean_ms": 10.0,
                "median_ms": 9.0,
                "p90_ms": 12.0,
                "p99_ms": 15.0,
                "min_ms": 8.0,
                "max_ms": 16.0,
            },
            "memory": {
                "inference_mb": 500.0,
                "peak_delta_mb": 600.0,
            },
        },
        "int8_ptq": {
            "model_size": {
                "total_bytes": 250,
                "total_mb": 0.25,
                "param_count": 250,
                "dtype_bytes": {"qwix:int8": 250},
            },
            "accuracy": {"loss": 1.05, "perplexity": 2.85765},
            "throughput": {"tokens_per_sec": 200.0, "avg_batch_time_ms": 5.0},
            "latency": {
                "mean_ms": 5.0,
                "median_ms": 4.5,
                "p90_ms": 6.0,
                "p99_ms": 7.5,
                "min_ms": 4.0,
                "max_ms": 8.0,
            },
            "memory": {
                "inference_mb": 250.0,
                "peak_delta_mb": 300.0,
            },
            "quantization_time_s": 12.5,
        },
    }


def test_save_json_preserves_dtype_bytes(tmp_path):
    path = tmp_path / "out.json"
    save_json(_sample_results(), str(path))
    payload = json.loads(path.read_text())
    assert "results" in payload
    base = payload["results"]["baseline"]
    # Nested dict must survive — this was the original bug.
    assert base["model_size"]["dtype_bytes"] == {"float32": 1000}
    assert payload["results"]["int8_ptq"]["model_size"]["dtype_bytes"] == {"qwix:int8": 250}


def test_save_json_includes_metadata(tmp_path):
    path = tmp_path / "out.json"
    save_json(_sample_results(), str(path), metadata={"foo": "bar"})
    payload = json.loads(path.read_text())
    assert payload["metadata"] == {"foo": "bar"}


def test_save_json_handles_jax_and_numpy_values(tmp_path):
    results = {
        "baseline": {
            "model_size": {"total_mb": np.float32(1.5)},
            "extra": {"arr": jnp.array([1, 2, 3])},
        }
    }
    path = tmp_path / "out.json"
    save_json(results, str(path))
    payload = json.loads(path.read_text())
    assert payload["results"]["baseline"]["model_size"]["total_mb"] == 1.5
    assert payload["results"]["baseline"]["extra"]["arr"] == [1, 2, 3]


def test_format_report_includes_baseline_deltas():
    text = format_report(_sample_results())
    # Compression: 1.0 / 0.25 = 4.00x
    assert "4.00x" in text
    # Latency speedup: 10 / 5 = 2.00x
    assert "2.00x" in text
    # Accuracy delta: (1.05 - 1.0) / 1.0 * 100 = +5.0%
    assert "+5.0%" in text
    assert "Quantization Time" in text


def test_format_report_with_metadata():
    text = format_report(_sample_results(), metadata={"timestamp": "2026-01-01T00:00:00"})
    assert "Run Metadata" in text
    assert "2026-01-01T00:00:00" in text


def test_format_report_markdown_emits_github_tables():
    md = format_report_markdown(_sample_results())
    assert md.startswith("# qwix quantization benchmark")
    assert "## Model Size" in md
    assert "|" in md  # table pipes
    assert "4.00x" in md


def test_save_csv_long_form(tmp_path):
    path = tmp_path / "out.csv"
    save_csv(_sample_results(), str(path))
    rows = list(csv.DictReader(path.open()))
    assert {"variant", "category", "metric", "value"} == set(rows[0].keys())
    # Should contain at least one row per (variant, category, metric).
    keys = {(r["variant"], r["category"], r["metric"]) for r in rows}
    assert ("baseline", "model_size", "total_bytes") in keys
    assert ("int8_ptq", "model_size", "dtype_bytes.qwix:int8") in keys
    assert ("int8_ptq", "quantization_time_s", "") in keys


def test_collect_metadata_basic():
    md = collect_metadata()
    assert "timestamp" in md
    assert "device_kind" in md
    assert "device_count" in md
    assert "versions" in md
    assert "jax" in md["versions"]
