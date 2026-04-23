"""Device memory (VRAM) measurement during inference.

Notes:
    JAX exposes ``device.memory_stats()`` on most accelerator backends. The
    ``peak_bytes_in_use`` counter is process-global and is not reset between
    calls, so absolute peaks across variants are not directly comparable on
    backends without a reset hook (CPU, some TPU configurations). For
    cross-variant comparison use ``peak_delta_bytes`` (per-call attribution)
    instead.
"""

from typing import Any

import jax
import jax.numpy as jnp


def _get_device_memory_stats() -> dict[str, int]:
    """Query memory stats from the first available JAX device."""
    devices = jax.devices()
    if not devices:
        return {}
    device = devices[0]
    try:
        mem = device.memory_stats() or {}
    except Exception:
        return {}
    return {
        "bytes_in_use": int(mem.get("bytes_in_use", 0)),
        "peak_bytes_in_use": int(mem.get("peak_bytes_in_use", 0)),
        "bytes_limit": int(mem.get("bytes_limit", 0)),
    }


def measure_memory(
    forward_fn: Any,
    sample_input: jax.Array | None = None,
) -> dict[str, Any]:
    """Measure peak device memory attributed to one forward pass.

    Args:
        forward_fn: Callable ``(tokens) -> logits``.
        sample_input: Input tensor; defaults to zeros ``(1, 128)``.

    Returns:
        Dict with ``"bytes_before"``, ``"peak_bytes"`` (raw global peak),
        ``"peak_delta_bytes"`` (peak attributable to this call),
        ``"inference_bytes"`` and ``"inference_mb"`` (alias of the delta).
    """
    if sample_input is None:
        sample_input = jnp.zeros((1, 128), dtype=jnp.int32)

    # Warmup: ensure compilation is done before we sample peaks.
    warm = forward_fn(sample_input)
    jax.block_until_ready(warm)

    before = _get_device_memory_stats()
    bytes_before = before.get("bytes_in_use", 0)
    peak_before = before.get("peak_bytes_in_use", 0)

    out = forward_fn(sample_input)
    jax.block_until_ready(out)

    after = _get_device_memory_stats()
    peak_bytes = after.get("peak_bytes_in_use", 0)

    peak_delta = max(peak_bytes - peak_before, 0)
    inference_bytes = max(peak_bytes - bytes_before, 0)

    return {
        "bytes_before": bytes_before,
        "peak_bytes": peak_bytes,
        "peak_delta_bytes": peak_delta,
        "peak_delta_mb": peak_delta / (1024 * 1024),
        "inference_bytes": inference_bytes,
        "inference_mb": inference_bytes / (1024 * 1024),
    }
