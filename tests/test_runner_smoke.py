"""End-to-end smoke test using a tiny synthetic Flax model.

We do NOT load MaxText here — instead we monkeypatch ``MaxTextModel.load`` to
return a hand-built Flax module so the rest of the runner pipeline (metrics
collection, reporting, JSON / Markdown / CSV serialisation) is exercised on
CPU in a few seconds.
"""

import json
from types import SimpleNamespace

import flax.linen as nn
import jax
import jax.numpy as jnp

from qwix_bench.config import BenchmarkConfig, QuantizationConfig
from qwix_bench.model import MaxTextModel
from qwix_bench.runner import BenchmarkRunner

VOCAB = 16
SEQ_LEN = 8


class TinyTransformer(nn.Module):
    """Minimal model with the same call signature as MaxText's transformer."""

    vocab_size: int = VOCAB

    @nn.compact
    def __call__(self, tokens, positions, segment_ids, enable_dropout=False):
        del positions, segment_ids, enable_dropout
        embed = nn.Embed(num_embeddings=self.vocab_size, features=8)(tokens)
        return nn.Dense(features=self.vocab_size)(embed)


def _fake_load(self):
    self.model = TinyTransformer()
    rng = jax.random.PRNGKey(0)
    sample = jnp.zeros((1, SEQ_LEN), dtype=jnp.int32)
    self.params = self.model.init(rng, sample, sample, sample)["params"]
    self.config = SimpleNamespace(vocab_size=VOCAB)
    self.mesh = None
    return self


def test_runner_smoke_baseline_only(monkeypatch, tmp_path):
    monkeypatch.setattr(MaxTextModel, "load", _fake_load)

    cfg = BenchmarkConfig(
        maxtext_args=[],
        quantizations=[],  # baseline-only keeps the test free of qwix internals.
        eval_seq_len=SEQ_LEN,
        warmup_steps=1,
        bench_steps=2,
        batch_size=1,
        run_accuracy=True,
        eval_tokens=[[1, 2, 3, 4, 5, 6, 7, 0]],
        output_file=str(tmp_path / "out.json"),
        output_markdown=str(tmp_path / "out.md"),
        output_csv=str(tmp_path / "out.csv"),
    )

    runner = BenchmarkRunner(cfg)
    results = runner.run()

    # Baseline metrics populated.
    assert "baseline" in results
    base = results["baseline"]
    for key in ("model_size", "accuracy", "memory", "throughput", "latency"):
        assert key in base, f"missing metric: {key}"

    # Outputs written.
    assert (tmp_path / "out.json").exists()
    assert (tmp_path / "out.md").exists()
    assert (tmp_path / "out.csv").exists()

    # JSON includes metadata + dtype_bytes.
    payload = json.loads((tmp_path / "out.json").read_text())
    assert "metadata" in payload
    assert "results" in payload
    assert "dtype_bytes" in payload["results"]["baseline"]["model_size"]

    # Reports produce non-empty strings.
    assert "QWIX QUANTIZATION BENCHMARK REPORT" in runner.report()
    assert "qwix quantization benchmark" in runner.report_markdown()


def test_runner_smoke_skips_disabled_metrics(monkeypatch):
    monkeypatch.setattr(MaxTextModel, "load", _fake_load)

    cfg = BenchmarkConfig(
        maxtext_args=[],
        quantizations=[],
        eval_seq_len=SEQ_LEN,
        warmup_steps=1,
        bench_steps=2,
        run_accuracy=False,
        run_memory=False,
        run_throughput=False,
        run_latency=False,
        run_model_size=True,
    )
    results = BenchmarkRunner(cfg).run()
    base = results["baseline"]
    assert "model_size" in base
    assert "accuracy" not in base
    assert "memory" not in base
    assert "throughput" not in base
    assert "latency" not in base


def test_runner_smoke_ptq_quantization(monkeypatch):
    """Run a real PTQ pass through qwix on the synthetic model."""
    monkeypatch.setattr(MaxTextModel, "load", _fake_load)

    cfg = BenchmarkConfig(
        maxtext_args=[],
        quantizations=[
            QuantizationConfig(name="int8_ptq", provider="ptq", weight_qtype="int8"),
        ],
        eval_seq_len=SEQ_LEN,
        warmup_steps=1,
        bench_steps=2,
        batch_size=1,
        run_accuracy=False,  # PTQ-quantized synthetic model; we don't assert on loss.
    )
    results = BenchmarkRunner(cfg).run()
    assert "baseline" in results
    assert "int8_ptq" in results
    assert "quantization_time_s" in results["int8_ptq"]
    assert results["int8_ptq"]["quantization_time_s"] >= 0
