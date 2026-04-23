"""Quantization utilities wrapping qwix.

Supports four providers via ``QuantizationConfig.provider``:

- ``"ptq"``  — Post-Training Quantization (no calibration)
- ``"qt"``   — Quantization-aware Training (no calibration; params left as-is)
- ``"awq"``  — Activation-aware Weight Quantization (needs calibration)
- ``"gptq"`` — Gradient-based PTQ (needs calibration; uses standard PTQ at inference)
"""

from typing import Any

import jax
import jax.numpy as jnp
import qwix
from qwix.contrib import awq, gptq

from qwix_bench.config import QuantizationConfig

# Mapping from string names to JAX / qwix dtypes.
_QTYPE_MAP: dict[str, Any] = {
    "int4": "int4",
    "int8": jnp.int8,
    "fp8": jnp.float8_e4m3fn,
    "nf4": "nf4",
}


def resolve_qtype(name: str | None) -> Any:
    """Map a config string (``"int8"``, ``"fp8"``, …) to a qwix-acceptable qtype."""
    if name is None:
        return None
    if name in _QTYPE_MAP:
        return _QTYPE_MAP[name]
    return getattr(jnp, name, name)


def _base_rule_kwargs(qcfg: QuantizationConfig) -> dict[str, Any]:
    weight_qtype = resolve_qtype(qcfg.weight_qtype)
    act_qtype = resolve_qtype(qcfg.act_qtype)

    kwargs: dict[str, Any] = {
        "module_path": qcfg.module_path,
        "weight_qtype": weight_qtype,
    }
    if act_qtype is not None:
        kwargs["act_qtype"] = act_qtype
    if qcfg.tile_size is not None:
        kwargs["tile_size"] = qcfg.tile_size
    if qcfg.act_calibration_method:
        kwargs["act_calibration_method"] = qcfg.act_calibration_method
    if qcfg.weight_calibration_method:
        kwargs["weight_calibration_method"] = qcfg.weight_calibration_method
    if qcfg.op_names is not None:
        kwargs["op_names"] = tuple(qcfg.op_names)
    return kwargs


def build_provider(qcfg: QuantizationConfig) -> qwix.QuantizationProvider:
    """Build a qwix provider for inference (post-quantization)."""
    rule_kwargs = _base_rule_kwargs(qcfg)
    if qcfg.provider == "ptq":
        return qwix.PtqProvider([qwix.QuantizationRule(**rule_kwargs)])
    if qcfg.provider == "qt":
        return qwix.QtProvider([qwix.QuantizationRule(**rule_kwargs)])
    if qcfg.provider == "gptq":
        # GPTQ params plug into a standard PtqProvider for inference.
        return qwix.PtqProvider([qwix.QuantizationRule(**rule_kwargs)])
    if qcfg.provider == "awq":
        rule = awq.AwqRule(**{**rule_kwargs, "n_grid": qcfg.n_grid})
        return awq.AwqInferenceProvider([rule])
    raise ValueError(f"Unknown provider type: {qcfg.provider!r}")


def _make_calibration_input(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    tokens = jax.random.randint(rng, (batch_size, seq_len), 0, vocab_size)
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
    segment_ids = jnp.ones_like(tokens)
    return tokens, positions, segment_ids


def _calibrate(
    cal_model: Any,
    variables: dict,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    n_batches: int,
    rng_seed: int,
) -> dict:
    """Run calibration forward passes; returns updated variables dict."""
    rng = jax.random.PRNGKey(rng_seed)
    for _ in range(n_batches):
        rng, data_rng = jax.random.split(rng)
        tokens, positions, segment_ids = _make_calibration_input(
            batch_size, seq_len, vocab_size, data_rng
        )
        _, updates = cal_model.apply(
            variables,
            tokens,
            positions,
            segment_ids,
            enable_dropout=False,
            mutable="quant_stats",
        )
        variables = {**variables, **updates}
    return variables


def _abstract_q_params(
    q_model: Any,
    *,
    batch_size: int,
    seq_len: int,
) -> Any:
    """Compute the abstract pytree shape of quantized params (used as a target)."""
    sample = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
    segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    abstract_vars = jax.eval_shape(
        q_model.init,
        jax.random.PRNGKey(99),
        sample,
        positions,
        segment_ids,
        enable_dropout=False,
    )
    return abstract_vars.get("params", abstract_vars)


def _params_subtree(params: Any) -> Any:
    if isinstance(params, dict) and "params" in params:
        return params["params"]
    return params


def quantize_model(
    model: Any,
    params: Any,
    qcfg: QuantizationConfig,
    *,
    batch_size: int = 1,
    seq_len: int = 512,
    vocab_size: int | None = None,
) -> tuple[Any, Any]:
    """Quantize a Flax model and its params using qwix.

    For ``ptq`` / ``qt`` no calibration is required. For ``awq`` / ``gptq``
    a synthetic-token calibration pass is performed first.

    Returns ``(quantized_model_for_inference, quantized_params)``.
    """
    raw_params = _params_subtree(params)
    variables = (
        params if isinstance(params, dict) and "params" in params else {"params": raw_params}
    )

    rule_kwargs = _base_rule_kwargs(qcfg)

    if qcfg.provider == "qt":
        # Quantized training keeps the params shape the same; no init needed.
        provider = qwix.QtProvider([qwix.QuantizationRule(**rule_kwargs)])
        return qwix.quantize_model(model, provider), raw_params

    if qcfg.provider == "ptq":
        provider = qwix.PtqProvider([qwix.QuantizationRule(**rule_kwargs)])
        q_model = qwix.quantize_model(model, provider)
        abstract_q_params = _abstract_q_params(q_model, batch_size=batch_size, seq_len=seq_len)
        q_params = qwix.quantize_params(raw_params, abstract_q_params)
        return q_model, q_params

    if vocab_size is None:
        raise ValueError(
            f"vocab_size is required for provider={qcfg.provider!r} (calibration needs synthetic tokens)"
        )

    if qcfg.provider == "gptq":
        gptq_rules = [gptq.GptqRule(**rule_kwargs)]
        cal_model = qwix.quantize_model(model, gptq.GptqCalibrationProvider(gptq_rules))
        cal_vars = _calibrate(
            cal_model,
            variables,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            n_batches=qcfg.calibration_batches,
            rng_seed=qcfg.rng_seed or 123,
        )
        ptq_provider = qwix.PtqProvider([qwix.QuantizationRule(**rule_kwargs)])
        ptq_model = qwix.quantize_model(model, ptq_provider)
        abstract_q_params = _abstract_q_params(ptq_model, batch_size=batch_size, seq_len=seq_len)
        q_params = gptq.quantize_params(
            params=cal_vars.get("params", raw_params),
            abstract_quantized_params=abstract_q_params,
            gptq_quant_stats=cal_vars.get("quant_stats", {}),
            gptq_block_size=qcfg.gptq_block_size,
            gptq_damping_factor=qcfg.gptq_damping,
            allow_extra_params=True,
        )
        return ptq_model, q_params

    if qcfg.provider == "awq":
        awq_rule_kwargs = {**rule_kwargs, "n_grid": qcfg.n_grid}
        awq_rules = [awq.AwqRule(**awq_rule_kwargs)]
        cal_model = qwix.quantize_model(model, awq.AwqCalibrationProvider(awq_rules))
        cal_vars = _calibrate(
            cal_model,
            variables,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            n_batches=qcfg.calibration_batches,
            rng_seed=qcfg.rng_seed or 456,
        )
        # Use a PTQ model only to derive the abstract param shape.
        ptq_provider = qwix.PtqProvider([qwix.QuantizationRule(**rule_kwargs)])
        ptq_model = qwix.quantize_model(model, ptq_provider)
        abstract_q_params = _abstract_q_params(ptq_model, batch_size=batch_size, seq_len=seq_len)
        q_params = awq.quantize_params(
            params=cal_vars.get("params", raw_params),
            abstract_quantized_params=abstract_q_params,
            awq_quant_stats=cal_vars.get("quant_stats", {}),
            n_grid=qcfg.n_grid,
            allow_extra_params=True,
        )
        inference_model = qwix.quantize_model(model, awq.AwqInferenceProvider(awq_rules))
        return inference_model, q_params

    raise ValueError(f"Unknown provider type: {qcfg.provider!r}")
