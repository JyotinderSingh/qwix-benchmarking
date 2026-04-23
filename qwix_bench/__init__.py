"""qwix_bench - Benchmarking framework for qwix quantization on MaxText models."""

from qwix_bench.config import VALID_PROVIDERS, BenchmarkConfig, QuantizationConfig
from qwix_bench.metrics import (
    measure_accuracy,
    measure_latency,
    measure_memory,
    measure_model_size,
    measure_throughput,
)
from qwix_bench.quantize import build_provider, quantize_model, resolve_qtype
from qwix_bench.report import (
    collect_metadata,
    format_report,
    format_report_markdown,
    save_csv,
    save_json,
)
from qwix_bench.runner import BenchmarkRunner


# MaxTextModel imports MaxText at module load; expose lazily to keep
# `from qwix_bench import ...` cheap for callers that only need the helpers.
def __getattr__(name: str):
    if name == "MaxTextModel":
        from qwix_bench.model import MaxTextModel as _M

        return _M
    raise AttributeError(name)


__all__ = [
    "VALID_PROVIDERS",
    "BenchmarkConfig",
    "BenchmarkRunner",
    "MaxTextModel",
    "QuantizationConfig",
    "build_provider",
    "collect_metadata",
    "format_report",
    "format_report_markdown",
    "measure_accuracy",
    "measure_latency",
    "measure_memory",
    "measure_model_size",
    "measure_throughput",
    "quantize_model",
    "resolve_qtype",
    "save_csv",
    "save_json",
]
