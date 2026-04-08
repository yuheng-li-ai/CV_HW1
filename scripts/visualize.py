"""Visualization entrypoint for curves, confusion matrix, weights, and errors."""

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
from cvhw1.visualization import Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--run-dir", required=True, help="Run directory containing history.json")
    parser.add_argument("--output-dir", default=None, help="Figure output directory")
    parser.add_argument("--tsne", action="store_true", help="Generate t-SNE figure")
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

    output_dir = Path(args.output_dir or Path(args.run_dir) / "figures")
    visualizer = Visualizer(dataset.class_names)
    visualizer.plot_history(Path(args.run_dir) / "history.json", output_dir)

    result = evaluator.evaluate(
        model,
        dataset.test,
        output_dir=output_dir,
        compute_tsne=args.tsne,
        tsne_sample_size=int(visualization_cfg.get("tsne_sample_size", 2000)),
    )
    visualizer.plot_confusion_matrix(result.confusion_matrix, output_dir / "confusion_matrix.png")
    visualizer.plot_first_layer_weights(
        model.layer1.weight.data,
        output_dir / "first_layer_weights.png",
        max_items=int(visualization_cfg.get("max_weight_maps", 64)),
    )
    errors = evaluator.collect_error_cases(
        dataset.test,
        result.predictions,
        max_items=int(visualization_cfg.get("max_error_cases", 16)),
    )
    visualizer.plot_error_cases(
        errors["images"],
        errors["true_labels"],
        errors["pred_labels"],
        output_dir / "error_cases.png",
        max_items=int(visualization_cfg.get("max_error_cases", 16)),
    )

    if args.tsne and result.embeddings is not None:
        labels = dataset.test.labels[: len(result.embeddings)]
        visualizer.plot_tsne(result.embeddings, labels, output_dir / "tsne_hidden_repr.png")

    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
