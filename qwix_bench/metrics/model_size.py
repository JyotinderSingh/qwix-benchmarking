"""Model size measurement."""

from typing import Any

import jax
import numpy as np
import qwix

# JAX rounds sub-byte dtypes up to 1 byte for ``itemsize``; track the real
# bit width so we account for packed storage correctly.
_SUB_BYTE_BITS: dict[str, int] = {
    "int4": 4,
    "uint4": 4,
    "nf4": 4,
}


def _dtype_bits(dtype: Any) -> int:
    """Return the storage bits for a JAX/numpy dtype, accounting for sub-byte types."""
    name = getattr(dtype, "name", str(dtype))
    if name in _SUB_BYTE_BITS:
        return _SUB_BYTE_BITS[name]
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return itemsize * 8
    return 8  # conservative default


def _array_bytes(size: int, dtype: Any) -> int:
    bits = _dtype_bits(dtype)
    return (size * bits + 7) // 8


def _qtype_bits(qtype: Any) -> int | None:
    """Return packed bits for a logical qtype (e.g. ``"int4"`` → 4)."""
    if qtype is None:
        return None
    name = getattr(qtype, "name", None) or str(qtype)
    return _SUB_BYTE_BITS.get(name)


def _leaf_bytes(leaf: Any) -> int:
    """Return the byte size of a single parameter leaf.

    Handles both regular arrays and qwix QArrays (including sub-byte qtypes
    like int4 / nf4 where storage is packed two-per-byte).
    """
    if isinstance(leaf, qwix.QArray):
        # qvalue: prefer logical qtype for packed-byte accounting.
        qbits = _qtype_bits(leaf.qtype) or _dtype_bits(leaf.qvalue.dtype)
        total = (leaf.qvalue.size * qbits + 7) // 8
        total += _array_bytes(leaf.scale.size, leaf.scale.dtype)
        if leaf.zero_point is not None:
            total += _array_bytes(leaf.zero_point.size, leaf.zero_point.dtype)
        return total
    if isinstance(leaf, jax.Array | np.ndarray):
        return _array_bytes(leaf.size, leaf.dtype)
    return 0


def measure_model_size(params: Any) -> dict[str, Any]:
    """Measure total parameter memory in bytes.

    Args:
        params: A pytree of model parameters (may contain QArrays).

    Returns:
        Dict with ``"total_bytes"``, ``"total_mb"``, ``"param_count"``,
        and per-dtype breakdown in ``"dtype_bytes"``.
    """
    leaves = jax.tree_util.tree_leaves(params, is_leaf=lambda x: isinstance(x, qwix.QArray))

    total_bytes = 0
    param_count = 0
    dtype_bytes: dict[str, int] = {}

    for leaf in leaves:
        nbytes = _leaf_bytes(leaf)
        if nbytes == 0:
            continue
        total_bytes += nbytes

        if isinstance(leaf, qwix.QArray):
            dtype_key = f"qwix:{leaf.qtype}"
            param_count += leaf.qvalue.size
        else:
            dtype_key = str(leaf.dtype)
            param_count += leaf.size

        dtype_bytes[dtype_key] = dtype_bytes.get(dtype_key, 0) + nbytes

    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "param_count": param_count,
        "dtype_bytes": dtype_bytes,
    }
