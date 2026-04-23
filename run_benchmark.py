#!/usr/bin/env python3
"""CLI entry point for qwix_bench."""

import argparse
import logging

from qwix_bench.config import BenchmarkConfig
from qwix_bench.runner import BenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark qwix quantization on MaxText models.",
    )
    parser.add_argument("config", help="Path to a YAML benchmark config file.")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to write JSON results (overrides config value).",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Path to write a Markdown report (overrides config value).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to write a long-form CSV (overrides config value).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = BenchmarkConfig.from_yaml(args.config)
    if args.output:
        cfg.output_file = args.output
    if args.markdown:
        cfg.output_markdown = args.markdown
    if args.csv:
        cfg.output_csv = args.csv

    runner = BenchmarkRunner(cfg)
    runner.run()
    print(runner.report())


if __name__ == "__main__":
    main()
