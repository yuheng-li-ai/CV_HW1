"""Training entrypoint for the Fashion-MNIST MLP homework."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cvhw1.data import load_fashion_mnist
from cvhw1.models import MLPClassifier
from cvhw1.training import Trainer
from cvhw1.utils import load_yaml, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["seed"]))
    dataset = load_fashion_mnist(
        data_dir=config["dataset"]["data_dir"],
        val_ratio=float(config["dataset"]["val_ratio"]),
        normalize=bool(config["dataset"].get("normalize", True)),
        seed=int(config["seed"]),
    )
    model = MLPClassifier(**config["model"])
    trainer = Trainer(config)
    output = trainer.fit(model, dataset.train, dataset.val)
    save_json(config, Path(output["artifacts"].run_dir) / "resolved_config.json")
    print(f"Best val accuracy: {output['metrics']['best_val_accuracy']:.4f}")
    print(f"Checkpoint: {output['artifacts'].checkpoint_path}")


if __name__ == "__main__":
    main()
