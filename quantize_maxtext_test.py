"""Self-contained absl test for quantizing MaxText models with qwix.

Given MaxText CLI args and a quantization recipe (name MaxText knows, e.g.
``int8`` / ``fp8`` / ``fp8_full`` / ``fp8_gpu`` / ``fp8_nanoo``), this script
loads a baseline MaxText model and a quantized variant and verifies that:

  1. The quantized forward pass runs end-to-end and returns correctly-shaped
     logits.
  2. The quantized logits actually differ from the baseline (otherwise the
     quantization path was a silent no-op).
  3. The quantized logits stay within a relative tolerance of the baseline.

The script uses MaxText's internal qwix integration (``use_qwix_quantization=
true``) rather than externally calling ``qwix.quantize_model``. The reason is
that MaxText's transformer is built from NNX modules wrapped by Linen via
``ToLinen``; without MaxText's internal ``_fix_for_qwix_quantization`` patching,
qwix's ``flax_util.find_param`` can't locate the kernels of those NNX layers
through the outer Linen scope, so external quantization becomes a no-op.

Test cases are parameterized — extend ``_RECIPES`` to cover more.

Usage (random-init):
    python quantize_maxtext_test.py --maxtext_model_name=default

Usage (real checkpoint):
    python quantize_maxtext_test.py \\
        --maxtext_model_name=llama3-8b \\
        --load_parameters_path=/path/to/checkpoint

Single case:
    python quantize_maxtext_test.py QuantizeMaxTextTest.test_quantize_int8

This file imports nothing from ``qwix_bench/``.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import jax
import jax.numpy as jnp
from absl import flags
from absl.testing import absltest, parameterized

from MaxText import maxtext_utils, pyconfig
from MaxText.layers import models, quantizations

_MAXTEXT_PKG_DIR = os.path.dirname(pyconfig.__file__)
_DEFAULT_BASE_CONFIG = os.path.join(_MAXTEXT_PKG_DIR, "configs", "base.yml")


# --------------------------------------------------------------------------- #
# Flags
# --------------------------------------------------------------------------- #

_BASE_CONFIG = flags.DEFINE_string(
    "maxtext_base_config",
    _DEFAULT_BASE_CONFIG,
    "Path to the MaxText base config YAML, forwarded as the first positional "
    "arg to pyconfig.initialize(). Defaults to the installed package's "
    "configs/base.yml so the script works without cd'ing into MaxText.",
)
_MODEL_NAME = flags.DEFINE_string(
    "maxtext_model_name",
    "default",
    "MaxText model_name= override (e.g. 'llama3-8b').",
)
_LOAD_PARAMETERS_PATH = flags.DEFINE_string(
    "load_parameters_path",
    None,
    "Optional checkpoint path. If unset, parameters are randomly initialized.",
)
_PER_DEVICE_BATCH_SIZE = flags.DEFINE_integer(
    "per_device_batch_size", 1, "MaxText per_device_batch_size override."
)
_MAX_TARGET_LENGTH = flags.DEFINE_integer(
    "max_target_length", 128, "MaxText max_target_length override."
)
_EXTRA_MAXTEXT_ARGS = flags.DEFINE_list(
    "extra_maxtext_args",
    [],
    "Comma-separated additional 'k=v' args appended verbatim to maxtext_args.",
)


# --------------------------------------------------------------------------- #
# Recipe
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class QuantizationRecipe:
    """Recipe routed through MaxText's internal qwix integration.

    ``maxtext_quantization`` is the value MaxText accepts under its
    ``quantization=`` config (see ``MaxText/layers/quantizations.py``).
    Supported: ``int8``, ``fp8``, ``fp8_full``, ``fp8_gpu``, ``fp8_nanoo``.
    """

    name: str
    maxtext_quantization: str


# --------------------------------------------------------------------------- #
# MaxText loading + forward pass
# --------------------------------------------------------------------------- #


def load_maxtext_model(
    maxtext_args: list[str],
    *,
    recipe: QuantizationRecipe | None = None,
) -> tuple[Any, Any, int]:
    """Initialize MaxText, build the model, and load decode params.

    If ``recipe`` is set, MaxText's internal qwix integration is enabled
    (``use_qwix_quantization=true``, ``quantization=<recipe>``) and the
    returned model is quantized via ``maybe_quantize_model``.

    Returns ``(model, params, vocab_size)``.
    """
    args = list(maxtext_args)
    if recipe is not None:
        args.append("use_qwix_quantization=true")
        args.append(f"quantization={recipe.maxtext_quantization}")
    else:
        args.append("use_qwix_quantization=false")

    # pyconfig.initialize expects argv where argv[0] is the program name and
    # argv[1] is the config path; mirror that shape here.
    config = pyconfig.initialize(["quantize_maxtext_test.py", *args])
    devices_array = maxtext_utils.create_device_mesh(config=config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

    quant = quantizations.configure_quantization(config)
    model = models.transformer_as_linen(config, mesh=mesh, quant=quant)
    if recipe is not None:
        model = quantizations.maybe_quantize_model(model, config)

    rng = jax.random.PRNGKey(0)
    state, _ = maxtext_utils.setup_decode_state(model, config, rng, mesh, None)
    return model, state.params, config.vocab_size


def forward_pass(model: Any, params: Any, tokens: jax.Array) -> jax.Array:
    """Run a single forward pass and return raw logits."""
    batch, seq_len = tokens.shape
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch, seq_len))
    segment_ids = jnp.ones_like(tokens)
    variables = (
        params
        if isinstance(params, dict) and "params" in params
        else {"params": params}
    )
    return model.apply(
        variables,
        tokens,
        positions,
        segment_ids,
        enable_dropout=False,
    )


def _build_maxtext_args() -> list[str]:
    args: list[str] = [
        _BASE_CONFIG.value,
        f"model_name={_MODEL_NAME.value}",
        f"per_device_batch_size={_PER_DEVICE_BATCH_SIZE.value}",
        f"max_target_length={_MAX_TARGET_LENGTH.value}",
        "skip_jax_distributed_system=true",
    ]
    if _LOAD_PARAMETERS_PATH.value:
        args.append(f"load_parameters_path={_LOAD_PARAMETERS_PATH.value}")
    args.extend(_EXTRA_MAXTEXT_ARGS.value)
    return args


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


_RECIPES: list[dict[str, Any]] = [
    dict(
        testcase_name="int8",
        recipe=QuantizationRecipe(name="int8", maxtext_quantization="int8"),
    ),
    dict(
        testcase_name="fp8",
        recipe=QuantizationRecipe(name="fp8", maxtext_quantization="fp8"),
    ),
    # NOTE: "fp8_full" is intentionally disabled — MaxText's get_quantization_rule
    # passes a `bwd_use_original_residuals` kwarg that the installed qwix.QtRule
    # doesn't accept (API drift between MaxText and qwix). Re-enable once the
    # versions are aligned.
    # dict(
    #     testcase_name="fp8_full",
    #     recipe=QuantizationRecipe(
    #         name="fp8_full", maxtext_quantization="fp8_full"
    #     ),
    # ),
]


class QuantizeMaxTextTest(parameterized.TestCase):
    """Validate that each recipe quantizes the configured MaxText model."""

    baseline_model: Any
    baseline_params: Any
    vocab_size: int
    batch: int
    seq_len: int
    tokens: jax.Array
    baseline_logits: jax.Array

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        maxtext_args = _build_maxtext_args()
        model, params, vocab_size = load_maxtext_model(maxtext_args)
        cls.baseline_model = model
        cls.baseline_params = params
        cls.vocab_size = vocab_size
        cls.batch = 1
        cls.seq_len = min(_MAX_TARGET_LENGTH.value, 64)

        cls.tokens = jax.random.randint(
            jax.random.PRNGKey(0), (cls.batch, cls.seq_len), 0, vocab_size
        )
        cls.baseline_logits = jax.jit(
            lambda toks: forward_pass(model, params, toks)
        )(cls.tokens)

    @parameterized.named_parameters(*_RECIPES)
    def test_quantize(self, recipe: QuantizationRecipe) -> None:
        maxtext_args = _build_maxtext_args()
        q_model, q_params, q_vocab_size = load_maxtext_model(
            maxtext_args, recipe=recipe
        )
        self.assertEqual(q_vocab_size, self.vocab_size)

        logits = jax.jit(lambda toks: forward_pass(q_model, q_params, toks))(
            self.tokens
        )
        self.assertEqual(
            logits.shape, (self.batch, self.seq_len, self.vocab_size)
        )

        max_abs_diff = float(jnp.max(jnp.abs(logits - self.baseline_logits)))
        mean_abs_diff = float(jnp.mean(jnp.abs(logits - self.baseline_logits)))
        baseline_abs_max = float(jnp.max(jnp.abs(self.baseline_logits)))
        print(
            f"[{recipe.name}] baseline |max|={baseline_abs_max:.4f}  "
            f"max|delta|={max_abs_diff:.4f}  mean|delta|={mean_abs_diff:.4f}"
        )

        # Quantization must actually have an effect — a zero delta means the
        # quantization path silently no-op'd.
        self.assertGreater(
            max_abs_diff,
            0.0,
            f"quantization had no effect on logits for {recipe.name}",
        )
        # …but it must stay within a relative tolerance of the baseline.
        self.assertLess(
            max_abs_diff,
            0.5 * max(baseline_abs_max, 1.0),
            f"quantized logits diverged too far from baseline for {recipe.name}",
        )


if __name__ == "__main__":
    absltest.main()
