"""Tests for measure_accuracy."""

import math

import jax.numpy as jnp

from qwix_bench.metrics.accuracy import measure_accuracy


def test_perfect_predictions_have_near_zero_loss():
    vocab = 5
    seq = 4
    targets = [0, 1, 2, 3]

    # Build logits with a huge spike on the next-token target so loss ~ 0.
    def fwd(tokens):
        # tokens shape (1, seq); we want logits[:, t-1, target_t] huge.
        logits = jnp.full((1, seq, vocab), -50.0)
        for t in range(1, seq):
            logits = logits.at[0, t - 1, targets[t]].set(50.0)
        return logits

    out = measure_accuracy(fwd, eval_tokens=[targets], seq_len=seq)
    assert out["loss"] < 0.01
    assert out["perplexity"] < 1.1


def test_uniform_logits_gives_log_vocab_loss():
    vocab = 8
    seq = 16
    tokens = [i % vocab for i in range(seq)]  # all targets in range

    def fwd(_tokens):
        return jnp.zeros((1, seq, vocab))

    out = measure_accuracy(fwd, eval_tokens=[tokens], seq_len=seq)
    assert math.isclose(out["loss"], math.log(vocab), rel_tol=1e-5)
    assert math.isclose(out["perplexity"], vocab, rel_tol=1e-4)


def test_padding_truncation():
    # eval token sequences shorter than seq_len should be padded.
    def fwd(tokens):
        assert tokens.shape == (1, 8)
        return jnp.zeros((1, 8, 4))

    out = measure_accuracy(fwd, eval_tokens=[[1, 2, 3]], seq_len=8)
    assert "loss" in out

    # Sequences longer than seq_len should be truncated.
    def fwd2(tokens):
        assert tokens.shape == (1, 4)
        return jnp.zeros((1, 4, 4))

    out2 = measure_accuracy(fwd2, eval_tokens=[[1, 2, 3, 4, 5, 6, 7]], seq_len=4)
    assert "loss" in out2
