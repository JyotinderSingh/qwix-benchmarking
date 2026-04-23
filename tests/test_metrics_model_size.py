"""Tests for measure_model_size."""

import jax.numpy as jnp
import qwix

from qwix_bench.metrics.model_size import measure_model_size


def test_plain_float32_array():
    params = {"dense": {"kernel": jnp.zeros((10, 20), dtype=jnp.float32)}}
    info = measure_model_size(params)
    assert info["total_bytes"] == 10 * 20 * 4
    assert info["param_count"] == 200
    assert info["dtype_bytes"] == {"float32": 800}
    assert abs(info["total_mb"] - 800 / (1024 * 1024)) < 1e-9


def test_mixed_dtypes():
    params = {
        "a": jnp.zeros((4, 4), dtype=jnp.float32),  # 64 bytes
        "b": jnp.zeros((4, 4), dtype=jnp.bfloat16),  # 32 bytes
    }
    info = measure_model_size(params)
    assert info["total_bytes"] == 96
    assert info["dtype_bytes"]["float32"] == 64
    assert info["dtype_bytes"]["bfloat16"] == 32
    assert info["param_count"] == 32


def test_qarray_int8():
    qvalue = jnp.zeros((10, 20), dtype=jnp.int8)  # 200 bytes
    scale = jnp.zeros((1, 20), dtype=jnp.float32)  # 80 bytes
    qarr = qwix.QArray(qvalue=qvalue, scale=scale, qtype=jnp.int8)
    info = measure_model_size({"w": qarr})
    assert info["total_bytes"] == 200 + 80
    assert info["param_count"] == 200
    assert "qwix:" in next(iter(info["dtype_bytes"]))


def test_qarray_int4_packed_storage():
    # int4: 100 elements should occupy 50 bytes regardless of jnp itemsize.
    qvalue = jnp.zeros((10, 10), dtype=jnp.int8)
    scale = jnp.zeros((1, 10), dtype=jnp.float32)  # 40 bytes
    qarr = qwix.QArray(qvalue=qvalue, scale=scale, qtype="int4")
    info = measure_model_size({"w": qarr})
    # qvalue: 100 * 4 bits = 50 bytes; scale: 40 bytes
    assert info["total_bytes"] == 50 + 40


def test_qarray_with_zero_point():
    qvalue = jnp.zeros((4, 4), dtype=jnp.int8)
    scale = jnp.zeros((1, 4), dtype=jnp.float32)
    zp = jnp.zeros((1, 4), dtype=jnp.int8)
    qarr = qwix.QArray(qvalue=qvalue, scale=scale, zero_point=zp, qtype=jnp.int8)
    info = measure_model_size({"w": qarr})
    # qvalue 16 + scale 16 + zp 4 = 36
    assert info["total_bytes"] == 36
