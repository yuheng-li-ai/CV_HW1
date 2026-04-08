# CV HW1: Three-Layer Fashion-MNIST MLP from Scratch

This repository contains a from-scratch implementation of a three-layer multilayer perceptron for Fashion-MNIST classification. The project follows the homework constraint that PyTorch, TensorFlow, JAX, and any other framework with built-in automatic differentiation must not be used. Forward propagation, backpropagation, optimization, checkpoint selection, evaluation, and visualization are implemented with NumPy and standard scientific Python utilities.

## Features

- Three-layer MLP with manually implemented backpropagation
- Configurable hidden dimension
- Switchable activation functions:
  `relu`, `sigmoid`, `tanh`, `leaky_relu`, `elu`, `softplus`, `swish`
- SGD optimizer with L2 regularization
- Learning-rate schedulers:
  `none`, `step`, `exponential`, `cosine`
- Validation-based checkpoint selection
- Hyperparameter search entrypoints
- Evaluation pipeline with accuracy, confusion matrix, per-class accuracy, and t-SNE
- Visualization pipeline for training curves, first-layer weights, error cases, and hidden-feature projections

## Environment

Recommended Python version: `Python 3.11`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Repository Layout

```text
.
├── configs/
├── scripts/
├── src/cvhw1/
├── reports/
├── artifacts/
├── notes/
├── HW1_计算机视觉.pdf
└── HW1_作业解析与执行计划.md
```

## Final Model

The current final configuration is stored in [configs/final_model.yaml](/home/lyh/study/CV/configs/final_model.yaml):

- activation: `leaky_relu`
- hidden dimension: `512`
- learning rate: `0.3`
- weight decay: `1e-4`
- epochs: `200`
- scheduler: `step`
- step size: `50`
- gamma: `0.5`

This model achieved:

- best validation accuracy: `0.9132`
- best validation epoch: `81`
- test accuracy: `0.9006`

Training metrics are stored in [train_metrics.json](/home/lyh/study/CV/artifacts/runs/grid_002_activation-leaky_relu_hidden_dim-512_learning_rate-0.3_weight_decay-0.0001_epochs-200_name-step_step_size-50_gamma-0.5/train_metrics.json), and test metrics are stored in [evaluation_metrics.json](/home/lyh/study/CV/artifacts/runs/final_model/evaluation/evaluation_metrics.json).

## Main Commands

Train a configuration:

```bash
python scripts/train.py --config configs/baseline.yaml
```

Run a search configuration:

```bash
python scripts/search.py --config configs/search_hidden_dim.yaml
```

Evaluate the final checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/final_model.yaml \
  --checkpoint artifacts/checkpoints/grid_002_activation-leaky_relu_hidden_dim-512_learning_rate-0.3_weight_decay-0.0001_epochs-200_name-step_step_size-50_gamma-0.5/best_model.npz \
  --output-dir artifacts/runs/final_model/evaluation \
  --tsne
```

Generate final report figures:

```bash
python scripts/visualize.py \
  --config configs/final_model.yaml \
  --checkpoint artifacts/checkpoints/grid_002_activation-leaky_relu_hidden_dim-512_learning_rate-0.3_weight_decay-0.0001_epochs-200_name-step_step_size-50_gamma-0.5/best_model.npz \
  --run-dir artifacts/runs/grid_002_activation-leaky_relu_hidden_dim-512_learning_rate-0.3_weight_decay-0.0001_epochs-200_name-step_step_size-50_gamma-0.5 \
  --output-dir artifacts/runs/final_model/figures \
  --tsne
```

Run the 200-epoch ReLU vs. Leaky ReLU comparison:

```bash
python scripts/search.py --config configs/search_activation_long_200.yaml
```

## Experimental Summary

The project used a staged search procedure rather than a single uncontrolled grid.

### Learning Rate

Under the short-run ReLU setting, validation performance improved as the learning rate increased from `0.01` to `0.3`, while `1.0` collapsed to near-random behavior. This established `0.3` as the strongest candidate region.

### Activation Function

At `20` epochs, `ReLU` slightly outperformed the other nonlinearities and `Leaky ReLU` was the closest competitor. After extending training to `200` epochs with the stronger schedule, `Leaky ReLU` slightly surpassed `ReLU`, which motivated the final activation choice.

### Weight Decay

Mild regularization performed best. `1e-4` slightly outperformed both no regularization and stronger penalties, while `1e-3` degraded validation accuracy.

### Hidden Dimension

Validation performance improved from `64` to `512`, but the gain saturated beyond `512`. Additional experiments at `768` and `1024` did not surpass the `512`-dimensional model while substantially increasing training cost.

### Scheduler

At `20` epochs with the strongest short-run configuration, `step` and `cosine` were the two strongest schedulers, with `step` slightly ahead. The final long-run configuration retained `step` decay but used a longer interval (`50` epochs) to make the schedule meaningful over `200` epochs.

## Training Policy

This project does not use explicit early stopping. Training runs for a fixed number of epochs, and the best checkpoint is selected according to validation accuracy. This design satisfies the homework requirement of automatic best-checkpoint saving on the validation set while keeping the training pipeline simple and reproducible.

## Outputs

Training logs, checkpoints, evaluation results, and figures are written under `artifacts/`.

Typical outputs include:

- `artifacts/checkpoints/.../best_model.npz`
- `artifacts/runs/.../history.json`
- `artifacts/runs/.../train_metrics.json`
- `artifacts/runs/.../evaluation/evaluation_metrics.json`
- `artifacts/runs/.../figures/loss_curves.png`
- `artifacts/runs/.../figures/confusion_matrix.png`
- `artifacts/runs/.../figures/first_layer_weights.png`
- `artifacts/runs/.../figures/error_cases.png`
- `artifacts/runs/.../figures/tsne_hidden_repr.png`
- `artifacts/figures/` for organized comparison figures across experiments

## Report Assets

The final model figures are located in [artifacts/runs/final_model/figures](/home/lyh/study/CV/artifacts/runs/final_model/figures). Organized comparison figures are located under [artifacts/figures](/home/lyh/study/CV/artifacts/figures), including:

- learning-rate comparisons
- activation comparisons
- weight-decay comparisons
- hidden-dimension comparisons
- scheduler comparisons

## Assignment Boundary

- Do not use PyTorch, TensorFlow, JAX, or any framework with built-in autodiff.
- Use NumPy for numerical computation and implement backward propagation manually.

## Notes

- The report is written in English and stored under `reports/`.
- Final trained weights should be uploaded to an external host, and the report should include the download link.
