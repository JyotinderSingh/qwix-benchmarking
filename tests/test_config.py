"""Tests for BenchmarkConfig / QuantizationConfig."""

import textwrap

import pytest

from qwix_bench.config import VALID_PROVIDERS, BenchmarkConfig, QuantizationConfig


def test_quantization_config_defaults():
    q = QuantizationConfig()
    assert q.name == "int8_ptq"
    assert q.provider == "ptq"
    assert q.weight_qtype == "int8"
    assert q.calibration_batches == 8
    assert q.n_grid == 20
    assert q.gptq_block_size == 128
    assert q.gptq_damping == 0.01


def test_quantization_config_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        QuantizationConfig(provider="bogus")


@pytest.mark.parametrize("provider", VALID_PROVIDERS)
def test_all_valid_providers_accepted(provider):
    QuantizationConfig(provider=provider)


def test_benchmark_config_defaults():
    cfg = BenchmarkConfig()
    assert cfg.run_accuracy is True
    assert cfg.run_model_size is True
    assert cfg.warmup_steps == 3
    assert cfg.bench_steps == 10
    assert cfg.batch_size == 1
    assert cfg.quantizations == []
    assert cfg.output_file is None
    assert cfg.output_markdown is None
    assert cfg.output_csv is None


def test_from_yaml_roundtrip(tmp_path):
    yaml_text = textwrap.dedent(
        """
        model_name: tiny
        maxtext_args:
          - foo=bar
        eval_seq_len: 16
        warmup_steps: 1
        bench_steps: 2
        quantizations:
          - name: int8
            provider: ptq
            weight_qtype: int8
          - name: gptq8
            provider: gptq
            weight_qtype: int8
            gptq_block_size: 64
            gptq_damping: 0.05
            calibration_batches: 4
          - name: awq8
            provider: awq
            weight_qtype: int8
            n_grid: 30
            calibration_batches: 4
        """
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)

    cfg = BenchmarkConfig.from_yaml(str(path))
    assert cfg.model_name == "tiny"
    assert cfg.eval_seq_len == 16
    assert cfg.warmup_steps == 1
    assert cfg.bench_steps == 2
    assert cfg.source_path == str(path)

    assert len(cfg.quantizations) == 3
    ptq, gptq, awq = cfg.quantizations
    assert ptq.provider == "ptq"
    assert gptq.provider == "gptq"
    assert gptq.gptq_block_size == 64
    assert gptq.gptq_damping == 0.05
    assert awq.provider == "awq"
    assert awq.n_grid == 30


def test_from_yaml_unknown_top_level_key_raises(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("model_name: tiny\nbogus_key: 42\n")
    with pytest.raises(TypeError):
        BenchmarkConfig.from_yaml(str(path))


def test_from_yaml_empty_file(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("")
    cfg = BenchmarkConfig.from_yaml(str(path))
    assert cfg.quantizations == []
