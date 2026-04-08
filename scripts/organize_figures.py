"""Collect and organize figure assets into a cleaner report-facing directory."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cvhw1.data import CLASS_NAMES
from cvhw1.utils import load_json
from cvhw1.utils.plotting import save_curve_plot
from cvhw1.visualization import Visualizer


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def organize_baseline(figures_root: Path) -> None:
    baseline_root = figures_root / "baseline"
    curves_dir = baseline_root / "curves"
    evaluation_dir = baseline_root / "evaluation"
    analysis_dir = baseline_root / "analysis"
    source = ROOT / "artifacts" / "runs" / "baseline" / "figures"

    copy_if_exists(source / "loss_curves.png", curves_dir / "loss_curves.png")
    copy_if_exists(source / "val_accuracy_curve.png", curves_dir / "val_accuracy_curve.png")
    copy_if_exists(source / "learning_rate_curve.png", curves_dir / "learning_rate_curve.png")
    copy_if_exists(source / "confusion_matrix.png", evaluation_dir / "confusion_matrix.png")
    copy_if_exists(source / "first_layer_weights.png", analysis_dir / "first_layer_weights.png")
    copy_if_exists(source / "error_cases.png", analysis_dir / "error_cases.png")
    copy_if_exists(source / "tsne_hidden_repr.png", analysis_dir / "tsne_hidden_repr.png")


def _lr_run_name(lr: str) -> str:
    mapping = {
        "0.01": "grid_001_activation-relu_hidden_dim-128_learning_rate-0.01_weight_decay-0.0005",
        "0.05": "grid_002_activation-relu_hidden_dim-128_learning_rate-0.05_weight_decay-0.0005",
        "0.1": "grid_003_activation-relu_hidden_dim-128_learning_rate-0.1_weight_decay-0.0005",
        "0.15": "grid_004_activation-relu_hidden_dim-128_learning_rate-0.15_weight_decay-0.0005",
        "0.2": "grid_005_activation-relu_hidden_dim-128_learning_rate-0.2_weight_decay-0.0005",
        "0.3": "grid_006_activation-relu_hidden_dim-128_learning_rate-0.3_weight_decay-0.0005",
    }
    return mapping[lr]


def organize_lr_comparison(figures_root: Path) -> None:
    comparison_root = figures_root / "lr_comparison"
    visualizer = Visualizer(CLASS_NAMES)

    lr_to_history = {}
    for lr in ("0.01", "0.05", "0.1", "0.15", "0.2", "0.3"):
        run_dir = ROOT / "artifacts" / "runs" / _lr_run_name(lr)
        history_path = run_dir / "history.json"
        if not history_path.exists():
            continue
        trial_dir = comparison_root / f"lr_{lr}"
        visualizer.plot_history(history_path, trial_dir)
        lr_to_history[lr] = load_json(history_path)

    if not lr_to_history:
        return

    save_curve_plot(
        {f"lr={lr}": history["train_loss"] for lr, history in lr_to_history.items()},
        title="Training Loss Comparison Across Learning Rates",
        ylabel="Training Loss",
        path=comparison_root / "train_loss_comparison.png",
    )
    save_curve_plot(
        {f"lr={lr}": history["val_loss"] for lr, history in lr_to_history.items()},
        title="Validation Loss Comparison Across Learning Rates",
        ylabel="Validation Loss",
        path=comparison_root / "val_loss_comparison.png",
    )
    save_curve_plot(
        {f"lr={lr}": history["val_accuracy"] for lr, history in lr_to_history.items()},
        title="Validation Accuracy Comparison Across Learning Rates",
        ylabel="Validation Accuracy",
        path=comparison_root / "val_accuracy_comparison.png",
    )


def organize_activation_comparison(figures_root: Path) -> None:
    comparison_root = figures_root / "activation_comparison"
    visualizer = Visualizer(CLASS_NAMES)
    activation_to_run = {
        "relu": "grid_001_activation-relu_hidden_dim-128_learning_rate-0.3_weight_decay-0.0005",
        "leaky_relu": "grid_002_activation-leaky_relu_hidden_dim-128_learning_rate-0.3_weight_decay-0.0005",
        "elu": "grid_003_activation-elu_hidden_dim-128_learning_rate-0.3_weight_decay-0.0005",
        "sigmoid": "grid_004_activation-sigmoid_hidden_dim-128_learning_rate-0.3_weight_decay-0.0005",
        "tanh": "grid_005_activation-tanh_hidden_dim-128_learning_rate-0.3_weight_decay-0.0005",
        "softplus": "grid_006_activation-softplus_hidden_dim-128_learning_rate-0.3_weight_decay-0.0005",
        "swish": "grid_007_activation-swish_hidden_dim-128_learning_rate-0.3_weight_decay-0.0005",
    }
    activation_to_history = {}
    for activation, run_name in activation_to_run.items():
        run_dir = ROOT / "artifacts" / "runs" / run_name
        history_path = run_dir / "history.json"
        if not history_path.exists():
            continue
        trial_dir = comparison_root / activation
        visualizer.plot_history(history_path, trial_dir)
        activation_to_history[activation] = load_json(history_path)

    if not activation_to_history:
        return

    save_curve_plot(
        {activation: history["train_loss"] for activation, history in activation_to_history.items()},
        title="Training Loss Comparison Across Activations",
        ylabel="Training Loss",
        path=comparison_root / "train_loss_comparison.png",
    )
    save_curve_plot(
        {activation: history["val_loss"] for activation, history in activation_to_history.items()},
        title="Validation Loss Comparison Across Activations",
        ylabel="Validation Loss",
        path=comparison_root / "val_loss_comparison.png",
    )
    save_curve_plot(
        {activation: history["val_accuracy"] for activation, history in activation_to_history.items()},
        title="Validation Accuracy Comparison Across Activations",
        ylabel="Validation Accuracy",
        path=comparison_root / "val_accuracy_comparison.png",
    )


def organize_weight_decay_comparison(figures_root: Path) -> None:
    comparison_root = figures_root / "weight_decay_comparison"
    visualizer = Visualizer(CLASS_NAMES)
    wd_to_run = {
        "0.0": "grid_001_activation-relu_hidden_dim-128_learning_rate-0.3_weight_decay-0.0",
        "1e-4": "grid_002_activation-relu_hidden_dim-128_learning_rate-0.3_weight_decay-0.0001",
        "5e-4": "grid_003_activation-relu_hidden_dim-128_learning_rate-0.3_weight_decay-0.0005",
        "1e-3": "grid_004_activation-relu_hidden_dim-128_learning_rate-0.3_weight_decay-0.001",
    }
    wd_to_history = {}
    for weight_decay, run_name in wd_to_run.items():
        run_dir = ROOT / "artifacts" / "runs" / run_name
        history_path = run_dir / "history.json"
        if not history_path.exists():
            continue
        trial_dir = comparison_root / f"wd_{weight_decay}"
        visualizer.plot_history(history_path, trial_dir)
        wd_to_history[weight_decay] = load_json(history_path)

    if not wd_to_history:
        return

    save_curve_plot(
        {f"wd={wd}": history["train_loss"] for wd, history in wd_to_history.items()},
        title="Training Loss Comparison Across Weight Decay Values",
        ylabel="Training Loss",
        path=comparison_root / "train_loss_comparison.png",
    )
    save_curve_plot(
        {f"wd={wd}": history["val_loss"] for wd, history in wd_to_history.items()},
        title="Validation Loss Comparison Across Weight Decay Values",
        ylabel="Validation Loss",
        path=comparison_root / "val_loss_comparison.png",
    )
    save_curve_plot(
        {f"wd={wd}": history["val_accuracy"] for wd, history in wd_to_history.items()},
        title="Validation Accuracy Comparison Across Weight Decay Values",
        ylabel="Validation Accuracy",
        path=comparison_root / "val_accuracy_comparison.png",
    )


def organize_hidden_dim_comparison(figures_root: Path) -> None:
    comparison_root = figures_root / "hidden_dim_comparison"
    visualizer = Visualizer(CLASS_NAMES)
    dim_to_run = {
        "64": "grid_001_activation-relu_hidden_dim-64_learning_rate-0.3_weight_decay-0.0001",
        "128": "grid_002_activation-relu_hidden_dim-128_learning_rate-0.3_weight_decay-0.0001",
        "256": "grid_001_activation-relu_hidden_dim-256_learning_rate-0.3_weight_decay-0.0001",
        "512": "grid_002_activation-relu_hidden_dim-512_learning_rate-0.3_weight_decay-0.0001",
        "768": "grid_003_activation-relu_hidden_dim-768_learning_rate-0.3_weight_decay-0.0001",
        "1024": "grid_004_activation-relu_hidden_dim-1024_learning_rate-0.3_weight_decay-0.0001",
    }
    dim_to_history = {}
    for hidden_dim, run_name in dim_to_run.items():
        run_dir = ROOT / "artifacts" / "runs" / run_name
        history_path = run_dir / "history.json"
        if not history_path.exists():
            continue
        trial_dir = comparison_root / f"dim_{hidden_dim}"
        visualizer.plot_history(history_path, trial_dir)
        dim_to_history[hidden_dim] = load_json(history_path)

    if not dim_to_history:
        return

    save_curve_plot(
        {f"dim={dim}": history["train_loss"] for dim, history in dim_to_history.items()},
        title="Training Loss Comparison Across Hidden Dimensions",
        ylabel="Training Loss",
        path=comparison_root / "train_loss_comparison.png",
    )
    save_curve_plot(
        {f"dim={dim}": history["val_loss"] for dim, history in dim_to_history.items()},
        title="Validation Loss Comparison Across Hidden Dimensions",
        ylabel="Validation Loss",
        path=comparison_root / "val_loss_comparison.png",
    )
    save_curve_plot(
        {f"dim={dim}": history["val_accuracy"] for dim, history in dim_to_history.items()},
        title="Validation Accuracy Comparison Across Hidden Dimensions",
        ylabel="Validation Accuracy",
        path=comparison_root / "val_accuracy_comparison.png",
    )


def organize_scheduler_comparison(figures_root: Path) -> None:
    comparison_root = figures_root / "scheduler_comparison"
    visualizer = Visualizer(CLASS_NAMES)
    scheduler_to_run = {
        "none": "grid_001_activation-relu_hidden_dim-512_learning_rate-0.3_weight_decay-0.0001_name-none",
        "step": "grid_002_activation-relu_hidden_dim-512_learning_rate-0.3_weight_decay-0.0001_name-step",
        "exponential": "grid_003_activation-relu_hidden_dim-512_learning_rate-0.3_weight_decay-0.0001_name-exponential",
        "cosine": "grid_004_activation-relu_hidden_dim-512_learning_rate-0.3_weight_decay-0.0001_name-cosine",
    }
    scheduler_to_history = {}
    for scheduler, run_name in scheduler_to_run.items():
        run_dir = ROOT / "artifacts" / "runs" / run_name
        history_path = run_dir / "history.json"
        if not history_path.exists():
            continue
        trial_dir = comparison_root / scheduler
        visualizer.plot_history(history_path, trial_dir)
        scheduler_to_history[scheduler] = load_json(history_path)

    if not scheduler_to_history:
        return

    save_curve_plot(
        {scheduler: history["train_loss"] for scheduler, history in scheduler_to_history.items()},
        title="Training Loss Comparison Across Schedulers",
        ylabel="Training Loss",
        path=comparison_root / "train_loss_comparison.png",
    )
    save_curve_plot(
        {scheduler: history["val_loss"] for scheduler, history in scheduler_to_history.items()},
        title="Validation Loss Comparison Across Schedulers",
        ylabel="Validation Loss",
        path=comparison_root / "val_loss_comparison.png",
    )
    save_curve_plot(
        {scheduler: history["val_accuracy"] for scheduler, history in scheduler_to_history.items()},
        title="Validation Accuracy Comparison Across Schedulers",
        ylabel="Validation Accuracy",
        path=comparison_root / "val_accuracy_comparison.png",
    )


def main() -> None:
    figures_root = ROOT / "artifacts" / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    organize_baseline(figures_root)
    organize_lr_comparison(figures_root)
    organize_activation_comparison(figures_root)
    organize_weight_decay_comparison(figures_root)
    organize_hidden_dim_comparison(figures_root)
    organize_scheduler_comparison(figures_root)
    print(f"Organized figures written to: {figures_root}")


if __name__ == "__main__":
    main()
