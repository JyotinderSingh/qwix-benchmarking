# qwix_bench

Benchmarking framework for evaluating [qwix](../qwix/) quantization recipes
on [MaxText](../maxtext/) models.

## Metrics

| Metric | What it measures |
|---|---|
| **Accuracy** | Cross-entropy loss and perplexity before/after quantization (vs baseline Δ) |
| **Model size** | Total parameter memory (bytes), per-dtype breakdown, compression ratio |
| **Inference memory** | Peak device VRAM attributable to a single forward pass (Δ from baseline peak) |
| **Throughput** | Tokens processed per second + speedup vs baseline |
| **Latency** | Per-batch forward pass timing — mean, median, p90, p99 + speedup |
| **Quantization time** | Wall-clock seconds spent in calibration + quantization |

## Quick start

```bash
# Install (qwix and MaxText are expected to be installed alongside).
pip install -e .[dev]

# Run a benchmark
python run_benchmark.py configs/example.yaml -v
```

## Quantization providers

| Provider | Description | Calibration |
|---|---|---|
| `ptq` | Post-training quantization (round-to-nearest) | None |
| `qt`  | Quantization-aware training (params left as-is, model wrapped) | None |
| `awq` | Activation-aware Weight Quantization (per-channel salient scaling) | Synthetic tokens × `calibration_batches` |
| `gptq` | Gradient-based PTQ using Hessian information | Synthetic tokens × `calibration_batches` |

See [configs/awq_example.yaml](configs/awq_example.yaml) and
[configs/gptq_example.yaml](configs/gptq_example.yaml).

## Configuration

Benchmarks are configured via YAML files. Key sections:

- **`maxtext_args`** — arguments forwarded to `MaxText.pyconfig.initialize()`
- **`quantizations`** — list of recipes to benchmark against the baseline
- **`run_*`** flags — toggle individual metrics on/off
- **`eval_tokens`** — token id sequences for accuracy evaluation
- **`output_file` / `output_markdown` / `output_csv`** — optional output paths (also overridable via CLI)

### Quantization recipe fields

| Field | Default | Description |
|---|---|---|
| `name` | `"int8_ptq"` | Human-readable label for the report |
| `provider` | `"ptq"` | One of `"ptq"`, `"qt"`, `"awq"`, `"gptq"` |
| `weight_qtype` | `"int8"` | Weight quantization type (`int4`, `int8`, `fp8`, `nf4`) |
| `act_qtype` | `None` | Activation quantization type (optional) |
| `module_path` | `".*"` | Regex selecting which modules to quantize |
| `tile_size` | `None` | Subchannel tile size |
| `calibration_batches` | `8` | Number of synthetic-token calibration batches (AWQ/GPTQ) |
| `n_grid` | `20` | AWQ grid-search size |
| `gptq_block_size` | `128` | GPTQ block size |
| `gptq_damping` | `0.01` | GPTQ damping factor |

## Programmatic usage

```python
from qwix_bench import BenchmarkConfig, BenchmarkRunner

cfg = BenchmarkConfig.from_yaml("configs/my_bench.yaml")
runner = BenchmarkRunner(cfg)
runner.run()
print(runner.report())
print(runner.report_markdown())
```

Run individual metrics directly:

```python
from qwix_bench.metrics import measure_model_size, measure_latency

size_info = measure_model_size(params)
latency_info = measure_latency(forward_fn, batch_size=1, seq_len=512)
```

## Development

```bash
# Lint + format
ruff check . && ruff format --check .

# Tests (CPU-only, no MaxText required)
pytest -v
```

## Project structure

```
qwix_bench/
├── configs/
│   ├── example.yaml          # PTQ example
│   ├── awq_example.yaml      # AWQ example
│   └── gptq_example.yaml     # GPTQ example
├── qwix_bench/
│   ├── __init__.py            # Public API surface
│   ├── config.py              # BenchmarkConfig / QuantizationConfig dataclasses
│   ├── model.py               # MaxText model loading wrapper
│   ├── quantize.py            # qwix wrapper: PTQ / QT / AWQ / GPTQ
│   ├── runner.py              # Main orchestrator
│   ├── report.py              # Text / Markdown / JSON / CSV output + metadata
│   └── metrics/
│       ├── accuracy.py        # Sparse cross-entropy loss / perplexity
│       ├── model_size.py      # Parameter byte counts (sub-byte aware)
│       ├── memory.py          # Device VRAM usage (peak delta)
│       ├── throughput.py      # Tokens/sec
│       └── latency.py         # Per-batch timing statistics
├── tests/                      # CPU-only unit + smoke tests
├── run_benchmark.py            # CLI entry point
└── pyproject.toml
```
