"""Benchmark configuration."""

import dataclasses
from typing import Any

import yaml

VALID_PROVIDERS = ("ptq", "qt", "awq", "gptq")


@dataclasses.dataclass
class QuantizationConfig:
    """Configuration for a single qwix quantization recipe."""

    name: str = "int8_ptq"
    provider: str = "ptq"  # one of VALID_PROVIDERS
    weight_qtype: str = "int8"
    act_qtype: str | None = None
    module_path: str = ".*"
    tile_size: int | float | None = None
    act_calibration_method: str = "absmax"
    weight_calibration_method: str = "absmax"
    op_names: tuple[str, ...] | None = None

    # Calibration knobs (used by AWQ / GPTQ).
    calibration_batches: int = 8
    rng_seed: int = 0

    # AWQ-specific.
    n_grid: int = 20

    # GPTQ-specific.
    gptq_block_size: int = 128
    gptq_damping: float = 0.01

    def __post_init__(self) -> None:
        if self.provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Unknown provider {self.provider!r}; expected one of {VALID_PROVIDERS}"
            )


@dataclasses.dataclass
class BenchmarkConfig:
    """Top-level benchmark configuration."""

    # -- MaxText model --------------------------------------------------------
    model_name: str = ""
    maxtext_args: list[str] = dataclasses.field(default_factory=list)

    # -- Quantization recipes to benchmark ------------------------------------
    quantizations: list[QuantizationConfig] = dataclasses.field(default_factory=list)

    # -- Which metrics to run -------------------------------------------------
    run_accuracy: bool = True
    run_model_size: bool = True
    run_memory: bool = True
    run_throughput: bool = True
    run_latency: bool = True

    # -- Accuracy settings ----------------------------------------------------
    eval_tokens: list[list[int]] | None = None
    eval_seq_len: int = 512

    # -- Throughput / latency settings ----------------------------------------
    warmup_steps: int = 3
    bench_steps: int = 10
    batch_size: int = 1

    # -- Output ---------------------------------------------------------------
    output_file: str | None = None
    output_markdown: str | None = None
    output_csv: str | None = None

    # -- Provenance (populated by from_yaml) ----------------------------------
    source_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        quant_dicts = raw.pop("quantizations", []) or []
        quants = [QuantizationConfig(**q) for q in quant_dicts]

        cfg = cls(quantizations=quants, **raw)
        cfg.source_path = path
        return cfg
