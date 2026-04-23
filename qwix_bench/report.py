"""Results formatting and reporting.

Supports three output formats:

- ``format_report(results, metadata=...)``           — human-readable text
- ``format_report_markdown(results, metadata=...)``  — GitHub-flavored markdown
- ``save_json(results, path, metadata=...)``         — full structured dump
- ``save_csv(results, path)``                        — long-form tabular dump

A baseline-relative delta column is added to every metric where applicable;
the first variant in ``results`` is treated as the baseline.
"""

import csv
import datetime as _dt
import hashlib
import json
from collections.abc import Iterable
from typing import Any

import jax
import numpy as np
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        return f"{value:.4f}"
    return str(value)


def _delta_pct(baseline: float | None, candidate: float | None) -> str:
    if baseline is None or candidate is None or baseline == 0:
        return "—"
    pct = (candidate - baseline) / baseline * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def _ratio(baseline: float | None, candidate: float | None) -> str:
    if baseline is None or candidate is None or candidate == 0:
        return "—"
    return f"{baseline / candidate:.2f}x"


def _get(results: dict, variant: str, *path: str) -> Any:
    cur: Any = results.get(variant, {})
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _versions() -> dict[str, str]:
    out: dict[str, str] = {"jax": getattr(jax, "__version__", "unknown")}
    try:
        import qwix

        out["qwix"] = getattr(qwix, "__version__", "unknown")
    except ImportError:
        pass
    try:
        import flax

        out["flax"] = getattr(flax, "__version__", "unknown")
    except ImportError:
        pass
    return out


def collect_metadata(cfg: Any = None) -> dict[str, Any]:
    """Build a run-metadata block for inclusion in reports / JSON."""
    devices = jax.devices()
    md: dict[str, Any] = {
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "versions": _versions(),
        "device_kind": devices[0].platform if devices else "unknown",
        "device_count": len(devices),
    }
    if cfg is not None:
        source = getattr(cfg, "source_path", None)
        if source:
            md["source_path"] = source
        try:
            payload = json.dumps(_dataclass_asdict(cfg), sort_keys=True, default=str)
            md["config_hash"] = hashlib.sha256(payload.encode()).hexdigest()[:12]
        except Exception:
            pass
    return md


def _dataclass_asdict(obj: Any) -> Any:
    import dataclasses

    if dataclasses.is_dataclass(obj):
        return {f.name: _dataclass_asdict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list | tuple):
        return [_dataclass_asdict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_asdict(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Section builders (shared between text and markdown)
# ---------------------------------------------------------------------------


def _section_rows(
    results: dict[str, dict[str, Any]],
    category: str,
    columns: list[tuple[str, list[str], str]],
    baseline: str,
) -> tuple[list[str], list[list[str]]] | None:
    """Build (headers, rows) for a metric category.

    columns: list of (header_label, key_path, kind) where kind is one of
    {"value", "delta_pct", "speedup", "compression"}. ``baseline`` only matters
    for delta/speedup columns.
    """
    if not any(category in results[v] for v in results):
        return None

    headers = ["Variant"] + [c[0] for c in columns]
    rows: list[list[str]] = []
    for v in results:
        row: list[str] = [v]
        for _, key_path, kind in columns:
            value = _get(results, v, category, *key_path)
            base_value = _get(results, baseline, category, *key_path)
            if kind == "value":
                row.append(_fmt(value))
            elif kind == "delta_pct":
                row.append("baseline" if v == baseline else _delta_pct(base_value, value))
            elif kind == "speedup" or kind == "compression":
                row.append("baseline" if v == baseline else _ratio(base_value, value))
            else:
                row.append(_fmt(value))
        rows.append(row)
    return headers, rows


_SECTIONS: list[tuple[str, str, list[tuple[str, list[str], str]]]] = [
    (
        "Model Size",
        "model_size",
        [
            ("Size (MB)", ["total_mb"], "value"),
            ("Params", ["param_count"], "value"),
            ("Compression", ["total_mb"], "compression"),
        ],
    ),
    (
        "Accuracy",
        "accuracy",
        [
            ("Loss", ["loss"], "value"),
            ("Δ Loss", ["loss"], "delta_pct"),
            ("Perplexity", ["perplexity"], "value"),
            ("Δ PPL", ["perplexity"], "delta_pct"),
        ],
    ),
    (
        "Inference Memory",
        "memory",
        [
            ("Inference (MB)", ["inference_mb"], "value"),
            ("Peak Δ (MB)", ["peak_delta_mb"], "value"),
            ("Δ", ["inference_mb"], "delta_pct"),
        ],
    ),
    (
        "Throughput",
        "throughput",
        [
            ("Tokens/s", ["tokens_per_sec"], "value"),
            ("Avg Batch (ms)", ["avg_batch_time_ms"], "value"),
            ("Speedup", ["avg_batch_time_ms"], "speedup"),
        ],
    ),
    (
        "Latency",
        "latency",
        [
            ("Mean (ms)", ["mean_ms"], "value"),
            ("Median (ms)", ["median_ms"], "value"),
            ("P90 (ms)", ["p90_ms"], "value"),
            ("P99 (ms)", ["p99_ms"], "value"),
            ("Speedup (mean)", ["mean_ms"], "speedup"),
        ],
    ),
]


def _baseline_name(results: dict[str, dict[str, Any]]) -> str:
    if "baseline" in results:
        return "baseline"
    return next(iter(results), "baseline")


def _quant_time_rows(results: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows = []
    for v, m in results.items():
        if "quantization_time_s" in m:
            rows.append([v, _fmt(m["quantization_time_s"])])
    return rows


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------


def format_report(
    results: dict[str, dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Produce a human-readable comparison report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  QWIX QUANTIZATION BENCHMARK REPORT")
    lines.append("=" * 78)

    if metadata:
        lines.append("")
        lines.append("--- Run Metadata ---")
        for key, value in metadata.items():
            lines.append(f"  {key}: {value}")

    baseline = _baseline_name(results)

    for title, category, columns in _SECTIONS:
        section = _section_rows(results, category, columns, baseline)
        if section is None:
            continue
        headers, rows = section
        lines.append("")
        lines.append(f"--- {title} ---")
        lines.append(tabulate(rows, headers=headers, tablefmt="simple"))

    quant_rows = _quant_time_rows(results)
    if quant_rows:
        lines.append("")
        lines.append("--- Quantization Time ---")
        lines.append(tabulate(quant_rows, headers=["Variant", "Seconds"], tablefmt="simple"))

    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def format_report_markdown(
    results: dict[str, dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Produce a GitHub-flavored markdown report (paste-friendly for PRs)."""
    lines: list[str] = ["# qwix quantization benchmark"]

    if metadata:
        lines.append("")
        lines.append("## Run metadata")
        lines.append("")
        for key, value in metadata.items():
            lines.append(f"- **{key}**: `{value}`")

    baseline = _baseline_name(results)

    for title, category, columns in _SECTIONS:
        section = _section_rows(results, category, columns, baseline)
        if section is None:
            continue
        headers, rows = section
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append(tabulate(rows, headers=headers, tablefmt="github"))

    quant_rows = _quant_time_rows(results)
    if quant_rows:
        lines.append("")
        lines.append("## Quantization time")
        lines.append("")
        lines.append(tabulate(quant_rows, headers=["Variant", "Seconds"], tablefmt="github"))

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON / CSV serialisation
# ---------------------------------------------------------------------------


def _sanitise(value: Any) -> Any:
    """Convert values to JSON-serialisable form, preserving nested dicts/lists."""
    if isinstance(value, dict):
        return {str(k): _sanitise(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitise(v) for v in value]
    if isinstance(value, np.ndarray | jax.Array):
        return _sanitise(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return str(value)


def save_json(
    results: dict[str, dict[str, Any]],
    path: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write structured results (with metadata) to a JSON file."""
    payload = {
        "metadata": _sanitise(metadata) if metadata else {},
        "results": _sanitise(results),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _flatten(prefix: tuple[str, ...], value: Any) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _flatten((*prefix, str(k)), v)
    else:
        yield prefix, value


def save_csv(results: dict[str, dict[str, Any]], path: str) -> None:
    """Write results as a long-form CSV: one row per (variant, category, metric)."""
    rows: list[tuple[str, str, str, Any]] = []
    for variant, metrics in results.items():
        for category, values in metrics.items():
            if isinstance(values, dict):
                for path_keys, leaf in _flatten((), values):
                    rows.append((variant, category, ".".join(path_keys), _sanitise(leaf)))
            else:
                rows.append((variant, category, "", _sanitise(values)))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "category", "metric", "value"])
        for row in rows:
            writer.writerow(row)
