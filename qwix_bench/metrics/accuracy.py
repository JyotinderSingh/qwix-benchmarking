"""Accuracy metrics: cross-entropy loss and perplexity."""

from typing import Any

import jax
import jax.numpy as jnp


def _sparse_cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Token-level cross-entropy without materialising one-hot labels.

    Args:
        logits: ``(batch, seq_len, vocab_size)``
        targets: ``(batch, seq_len)`` integer token ids.

    Returns:
        Scalar mean loss over (batch * seq_len).
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return -jnp.mean(target_log_probs)


def measure_accuracy(
    forward_fn: Any,
    eval_tokens: list[list[int]],
    seq_len: int = 512,
) -> dict[str, float]:
    """Run evaluation and return loss / perplexity.

    Args:
        forward_fn: Callable ``(tokens: jax.Array) -> logits`` where
            tokens is ``(batch, seq_len)`` and logits is
            ``(batch, seq_len, vocab_size)``.
        eval_tokens: List of token-id sequences used for evaluation.
        seq_len: Sequences are truncated / padded to this length.

    Returns:
        Dict with ``"loss"`` and ``"perplexity"`` keys.
    """
    padded: list[list[int]] = []
    for seq in eval_tokens:
        if len(seq) >= seq_len:
            padded.append(seq[:seq_len])
        else:
            padded.append(seq + [0] * (seq_len - len(seq)))

    tokens = jnp.array(padded, dtype=jnp.int32)
    logits = forward_fn(tokens)

    # Shift so we predict next token: logits[:, :-1] -> targets[:, 1:].
    shift_logits = logits[:, :-1, :]
    shift_targets = tokens[:, 1:]

    loss = float(_sparse_cross_entropy(shift_logits, shift_targets))
    perplexity = float(jnp.exp(jnp.clip(loss, max=100.0)))

    return {"loss": loss, "perplexity": perplexity}
