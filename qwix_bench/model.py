"""Model loading utilities wrapping MaxText."""

from typing import Any

import jax
import jax.numpy as jnp
from MaxText import maxtext_utils, pyconfig
from MaxText.layers import models, quantizations


class MaxTextModel:
    """Wrapper around MaxText model loading for benchmarking.

    Handles config initialization, model creation, and parameter loading so
    that callers only need to supply MaxText CLI args.
    """

    def __init__(self, maxtext_args: list[str]):
        self.maxtext_args = maxtext_args
        self.config: Any = None
        self.params: Any = None
        self.model: Any = None
        self.mesh: Any = None

    # -- public API -----------------------------------------------------------

    def load(self) -> "MaxTextModel":
        """Initialize MaxText config, build the model, and load params."""
        argv = list(self.maxtext_args)
        self.config = pyconfig.initialize(argv)

        devices_array = maxtext_utils.create_device_mesh(config=self.config)
        self.mesh = jax.sharding.Mesh(devices_array, self.config.mesh_axes)

        quant = quantizations.configure_quantization(self.config)
        self.model = models.transformer_as_linen(
            self.config,
            mesh=self.mesh,
            quant=quant,
        )

        rng = jax.random.PRNGKey(0)
        state, _ = maxtext_utils.setup_decode_state(
            self.model,
            self.config,
            rng,
            self.mesh,
            None,
        )
        self.params = state.params
        return self

    def forward(self, tokens: jax.Array) -> jax.Array:
        """Run a forward pass (prefill) and return logits.

        ``tokens`` should be shape ``(batch, seq_len)`` of integer token ids.
        """
        if self.model is None:
            raise RuntimeError("Call .load() before .forward()")
        return forward_pass(self.model, self.params, tokens)

    def get_raw_model_and_params(self) -> tuple[Any, Any]:
        """Return the underlying Flax model and params for direct use."""
        if self.model is None:
            raise RuntimeError("Call .load() before accessing model/params")
        return self.model, self.params


def forward_pass(model: Any, params: Any, tokens: jax.Array) -> jax.Array:
    """Run a single forward pass returning raw logits.

    Builds positions and segment_ids inline so callers only need to provide
    token ids. Used by both the baseline and quantized variants.
    """
    batch, seq_len = tokens.shape
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch, seq_len))
    segment_ids = jnp.ones_like(tokens)

    variables = params if isinstance(params, dict) and "params" in params else {"params": params}
    return model.apply(
        variables,
        tokens,
        positions,
        segment_ids,
        enable_dropout=False,
    )


# Backwards-compat alias used by older callers / tests.
_forward = forward_pass
