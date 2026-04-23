"""Main benchmark runner that orchestrates model loading, quantization, and metrics."""

import functools
import logging
import time
from typing import Any

import jax
import jax.numpy as jnp

from qwix_bench.config import BenchmarkConfig
from qwix_bench.metrics import (
    measure_accuracy,
    measure_latency,
    measure_memory,
    measure_model_size,
    measure_throughput,
)
from qwix_bench.model import MaxTextModel, forward_pass
from qwix_bench.quantize import quantize_model
from qwix_bench.report import (
    collect_metadata,
    format_report,
    format_report_markdown,
    save_csv,
    save_json,
)

log = logging.getLogger(__name__)


def _make_jit_forward(model: Any, params: Any) -> Any:
    """Return a JIT-compiled callable ``(tokens) -> logits``."""
    fwd = functools.partial(forward_pass, model, params)
    return jax.jit(fwd)


class BenchmarkRunner:
    """Run quantization benchmarks for a MaxText model.

    Usage::

        cfg = BenchmarkConfig.from_yaml("configs/my_bench.yaml")
        runner = BenchmarkRunner(cfg)
        results = runner.run()
        print(runner.report())
    """

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.results: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, Any] = {}

    # -- public API -----------------------------------------------------------

    def run(self) -> dict[str, dict[str, Any]]:
        """Execute the full benchmark suite and return results."""
        log.info("Loading MaxText model...")
        mt = MaxTextModel(self.cfg.maxtext_args).load()
        model, params = mt.get_raw_model_and_params()
        vocab_size = getattr(mt.config, "vocab_size", None)

        # ---- Baseline ----
        log.info("Benchmarking baseline (unquantized) model...")
        baseline_fwd = _make_jit_forward(model, params)
        self.results["baseline"] = self._benchmark_variant(
            label="baseline",
            forward_fn=baseline_fwd,
            params=params,
        )

        # ---- Each quantization recipe ----
        for qcfg in self.cfg.quantizations:
            label = qcfg.name
            log.info("Benchmarking quantized variant: %s", label)

            t0 = time.perf_counter()
            q_model, q_params = quantize_model(
                model,
                params,
                qcfg,
                batch_size=self.cfg.batch_size,
                seq_len=self.cfg.eval_seq_len,
                vocab_size=vocab_size,
            )
            quant_time = time.perf_counter() - t0
            log.info("  [%s] quantization completed in %.1fs", label, quant_time)

            q_forward = _make_jit_forward(q_model, q_params)
            metrics = self._benchmark_variant(
                label=label,
                forward_fn=q_forward,
                params=q_params,
            )
            metrics["quantization_time_s"] = quant_time
            self.results[label] = metrics

        # ---- Capture metadata + outputs ----
        self.metadata = collect_metadata(self.cfg)

        if self.cfg.output_file:
            save_json(self.results, self.cfg.output_file, metadata=self.metadata)
            log.info("Results saved to %s", self.cfg.output_file)
        if self.cfg.output_markdown:
            with open(self.cfg.output_markdown, "w") as f:
                f.write(format_report_markdown(self.results, metadata=self.metadata))
            log.info("Markdown report saved to %s", self.cfg.output_markdown)
        if self.cfg.output_csv:
            save_csv(self.results, self.cfg.output_csv)
            log.info("CSV results saved to %s", self.cfg.output_csv)

        return self.results

    def report(self) -> str:
        """Return a formatted comparison report (text)."""
        return format_report(self.results, metadata=self.metadata or None)

    def report_markdown(self) -> str:
        """Return a markdown comparison report."""
        return format_report_markdown(self.results, metadata=self.metadata or None)

    # -- internals ------------------------------------------------------------

    def _benchmark_variant(
        self,
        label: str,
        forward_fn: Any,
        params: Any,
    ) -> dict[str, Any]:
        """Collect all enabled metrics for a single model variant."""
        results: dict[str, Any] = {}

        if self.cfg.run_model_size:
            log.info("  [%s] Measuring model size...", label)
            results["model_size"] = measure_model_size(params)

        if self.cfg.run_accuracy and self.cfg.eval_tokens:
            log.info("  [%s] Measuring accuracy...", label)
            results["accuracy"] = measure_accuracy(
                forward_fn,
                eval_tokens=self.cfg.eval_tokens,
                seq_len=self.cfg.eval_seq_len,
            )

        if self.cfg.run_memory:
            log.info("  [%s] Measuring inference memory...", label)
            sample = jnp.zeros((self.cfg.batch_size, self.cfg.eval_seq_len), dtype=jnp.int32)
            results["memory"] = measure_memory(forward_fn, sample_input=sample)

        if self.cfg.run_throughput:
            log.info("  [%s] Measuring throughput...", label)
            results["throughput"] = measure_throughput(
                forward_fn,
                batch_size=self.cfg.batch_size,
                seq_len=self.cfg.eval_seq_len,
                warmup_steps=self.cfg.warmup_steps,
                bench_steps=self.cfg.bench_steps,
            )

        if self.cfg.run_latency:
            log.info("  [%s] Measuring latency...", label)
            results["latency"] = measure_latency(
                forward_fn,
                batch_size=self.cfg.batch_size,
                seq_len=self.cfg.eval_seq_len,
                warmup_steps=self.cfg.warmup_steps,
                bench_steps=self.cfg.bench_steps,
            )

        return results
