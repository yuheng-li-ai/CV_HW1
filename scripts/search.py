"""Hyperparameter search entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cvhw1.search import HyperparameterSearcher
from cvhw1.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to search config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_config = load_yaml(args.config)
    base_config = load_yaml(search_config["base_config"])
    searcher = HyperparameterSearcher(base_config, search_config)
    summary = searcher.run()
    print("Best search result:")
    print(summary["best_result"])


if __name__ == "__main__":
    main()
