"""Grid-search orchestration."""

from __future__ import annotations

import itertools
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from cvhw1.data import load_fashion_mnist
from cvhw1.models import MLPClassifier
from cvhw1.training import Trainer
from cvhw1.utils.config import set_by_dotted_path
from cvhw1.utils.io import save_json
from cvhw1.utils.random import set_seed


class HyperparameterSearcher:
    def __init__(self, base_config: Dict[str, Any], search_config: Dict[str, Any]) -> None:
        self.base_config = base_config
        self.search_config = search_config

    def run(self) -> Dict[str, Any]:
        grid = self.search_config["grid"]
        keys = list(grid.keys())
        values = [grid[key] for key in keys]
        results = []
        best_result: Dict[str, Any] | None = None

        for trial_index, combination in enumerate(itertools.product(*values), start=1):
            trial_config = deepcopy(self.base_config)
            trial_name_parts = []
            for key, value in zip(keys, combination):
                set_by_dotted_path(trial_config, key, value)
                short_key = key.split(".")[-1]
                trial_name_parts.append(f"{short_key}-{value}")
            trial_config["logging"]["run_name"] = f"grid_{trial_index:03d}_" + "_".join(trial_name_parts)
            trial_config["checkpoint"]["save_dir"] = str(
                Path(self.base_config["checkpoint"]["save_dir"]).parent / trial_config["logging"]["run_name"]
            )

            set_seed(int(trial_config["seed"]))
            dataset = load_fashion_mnist(
                data_dir=trial_config["dataset"]["data_dir"],
                val_ratio=float(trial_config["dataset"]["val_ratio"]),
                normalize=bool(trial_config["dataset"].get("normalize", True)),
                seed=int(trial_config["seed"]),
            )
            model = MLPClassifier(**trial_config["model"])
            trainer = Trainer(trial_config)
            train_output = trainer.fit(model, dataset.train, dataset.val)
            result = {
                "trial_name": trial_config["logging"]["run_name"],
                "overrides": dict(zip(keys, combination)),
                "best_val_accuracy": train_output["metrics"]["best_val_accuracy"],
                "best_epoch": train_output["metrics"]["best_epoch"],
            }
            results.append(result)
            if best_result is None or result["best_val_accuracy"] > best_result["best_val_accuracy"]:
                best_result = result

        summary = {"best_result": best_result, "results": results}
        save_json(summary, Path(self.base_config["logging"]["output_dir"]) / "search_results.json")
        return summary
