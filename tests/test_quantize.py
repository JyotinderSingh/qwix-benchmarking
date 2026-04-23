"""Tests for build_provider and resolve_qtype."""

import jax.numpy as jnp
import pytest
import qwix
from qwix.contrib import awq

from qwix_bench.config import QuantizationConfig
from qwix_bench.quantize import build_provider, resolve_qtype


def test_resolve_qtype_known():
    assert resolve_qtype("int8") is jnp.int8
    assert resolve_qtype("fp8") is jnp.float8_e4m3fn
    assert resolve_qtype("int4") == "int4"
    assert resolve_qtype("nf4") == "nf4"
    assert resolve_qtype(None) is None


def test_resolve_qtype_falls_back_to_jnp():
    assert resolve_qtype("float32") is jnp.float32


def test_build_provider_ptq():
    p = build_provider(QuantizationConfig(provider="ptq", weight_qtype="int8"))
    assert isinstance(p, qwix.PtqProvider)


def test_build_provider_qt():
    p = build_provider(QuantizationConfig(provider="qt", weight_qtype="int8"))
    assert isinstance(p, qwix.QtProvider)


def test_build_provider_gptq_uses_ptq_for_inference():
    p = build_provider(QuantizationConfig(provider="gptq", weight_qtype="int8"))
    # GPTQ uses the standard PtqProvider for inference.
    assert isinstance(p, qwix.PtqProvider)


def test_build_provider_awq():
    p = build_provider(QuantizationConfig(provider="awq", weight_qtype="int8"))
    assert isinstance(p, awq.AwqInferenceProvider)


def test_build_provider_unknown_raises():
    # Bypass __post_init__ validation by mutating after construction.
    qcfg = QuantizationConfig(provider="ptq")
    qcfg.provider = "bogus"
    with pytest.raises(ValueError, match="Unknown provider"):
        build_provider(qcfg)
