"""Evaluation entrypoint for loading the best checkpoint and testing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cvhw1.data import load_fashion_mnist
from cvhw1.evaluation import Evaluator
from cvhw1.models import MLPClassifier
from cvhw1.utils import load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--output-dir", default=None, help="Directory for evaluation artifacts")
    parser.add_argument("--tsne", action="store_true", help="Generate t-SNE embeddings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    visualization_cfg = config.get("visualization", {})
    set_seed(int(config["seed"]))
    dataset = load_fashion_mnist(
        data_dir=config["dataset"]["data_dir"],
        val_ratio=float(config["dataset"]["val_ratio"]),
        normalize=bool(config["dataset"].get("normalize", True)),
        seed=int(config["seed"]),
    )
    model = MLPClassifier(**config["model"])
    evaluator = Evaluator(dataset.class_names)
    evaluator.load_checkpoint(model, args.checkpoint)
    output_dir = args.output_dir or Path(config["logging"]["output_dir"]) / "evaluation"
    result = evaluator.evaluate(
        model,
        dataset.test,
        output_dir=output_dir,
        compute_tsne=args.tsne,
        tsne_sample_size=int(visualization_cfg.get("tsne_sample_size", 2000)),
    )
    print(f"Test accuracy: {result.metrics['accuracy']:.4f}")
    print(f"Evaluation artifacts: {output_dir}")


if __name__ == "__main__":
    main()
