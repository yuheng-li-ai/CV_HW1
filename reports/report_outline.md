# Fashion-MNIST Classification with a Three-Layer Neural Network Implemented from Scratch

## Abstract

This report presents a from-scratch three-layer multilayer perceptron for Fashion-MNIST classification. The full training pipeline is implemented with NumPy, including forward propagation, backward propagation, loss computation, optimization, checkpoint selection, evaluation, and visualization. The study focuses on controlled comparisons of learning rate, activation function, weight decay, hidden dimension, and scheduler choice. Short-run experiments identify a strong optimization region, and a 200-epoch refinement stage is then used to select the final model. The final configuration uses hidden dimension 512, Leaky ReLU, initial learning rate 0.3, weight decay 1e-4, and step decay with a 50-epoch interval. This model reaches 0.9132 validation accuracy and 0.9006 test accuracy.

## 1. Introduction

The purpose of this project is to train a Fashion-MNIST classifier under the assignment constraint that all gradients and parameter updates must be implemented manually. The target model is intentionally simple: a three-layer MLP without convolutional structure or framework-based autodiff. This makes the experimental conclusions more directly attributable to optimization and representation choices. The main emphasis of the project is therefore not architectural novelty, but controlled analysis of how hyperparameters affect optimization quality and final accuracy.

## 2. Method

The classifier takes a 784-dimensional flattened input and maps it to ten output logits through two hidden layers and one output layer. Hidden width is configurable, and the nonlinearities can be switched across several activation functions. Training uses softmax cross-entropy with a numerically stabilized formulation and SGD with optional L2 regularization. The code does not use explicit early stopping. Instead, training proceeds for a fixed number of epochs and stores the checkpoint with the highest validation accuracy. This policy satisfies the assignment requirement of validation-based best-model saving while keeping the training loop simple and reproducible.

Fashion-MNIST images are normalized to the range [0, 1], and the original training split is further divided into training and validation subsets with a validation ratio of 0.1. All hyperparameter comparisons are based on validation performance, and the test set is used only after the final model is selected.

## 3. Experimental Setup

The experiments were conducted in two stages. The first stage used 20-epoch runs to screen candidate hyperparameters quickly. The second stage used longer training to verify whether the short-run ranking remained stable after more complete optimization.

The final selected configuration is shown in Table 1.

| Parameter | Value |
| --- | --- |
| Input dimension | 784 |
| Hidden dimension | 512 |
| Output dimension | 10 |
| Activation | Leaky ReLU |
| Batch size | 128 |
| Initial learning rate | 0.3 |
| Weight decay | 1e-4 |
| Scheduler | Step decay |
| Step size | 50 |
| Gamma | 0.5 |
| Number of epochs | 200 |
| Best validation epoch | 81 |

## 4. Results

### 4.1 Learning-Rate Comparison

The first question was whether the baseline learning rate was already large enough. Under the short-run ReLU setting, validation accuracy improved steadily as the initial learning rate increased from 0.01 to 0.3. This immediately showed that the original baseline range was too conservative. A second sweep then tested whether values above 0.3 would continue to improve the result. The answer was negative. A learning rate of 0.5 remained trainable but slightly weaker than 0.3, while 1.0 caused optimization collapse and produced accuracy close to random guessing. The useful region was therefore centered around 0.3 rather than around 0.05 or 0.1.

| Learning rate | Best validation accuracy | Best epoch |
| --- | --- | --- |
| 0.01 | 0.8478 | 16 |
| 0.05 | 0.8813 | 20 |
| 0.10 | 0.8928 | 20 |
| 0.15 | 0.8962 | 20 |
| 0.20 | 0.8988 | 19 |
| 0.30 | 0.9010 | 19 |
| 0.50 | 0.8950 | 19 |
| 1.00 | 0.1040 | 3 |

![Figure 1. Training-loss comparison across learning rates.](../../artifacts/figures/lr_comparison/train_loss_comparison.png)

![Figure 2. Validation-loss comparison across learning rates.](../../artifacts/figures/lr_comparison/val_loss_comparison.png)

![Figure 3. Validation-accuracy comparison across learning rates.](../../artifacts/figures/lr_comparison/val_accuracy_comparison.png)

The curves support the numerical result. Moderate increases in learning rate improved optimization speed and final validation accuracy, but the largest tested value destroyed stability. Observation: the useful learning-rate region extends well beyond the initial baseline. Explanation: the original step size underutilized the optimization budget, while 1.0 overshot the stable region. Conclusion: 0.3 is the strongest learning-rate candidate in the tested range.

### 4.2 Activation Comparison

The activation comparison was first performed under the 20-epoch setting to identify strong candidates efficiently. ReLU ranked first, but Leaky ReLU was already very close. Sigmoid was clearly the weakest activation, while Tanh, ELU, Swish, and Softplus formed a middle group with smaller differences.

| Activation | Best validation accuracy | Best epoch |
| --- | --- | --- |
| ReLU | 0.9010 | 19 |
| Leaky ReLU | 0.8998 | 17 |
| Tanh | 0.8958 | 17 |
| Swish | 0.8927 | 18 |
| ELU | 0.8917 | 20 |
| Softplus | 0.8800 | 16 |
| Sigmoid | 0.8565 | 20 |

![Figure 4. Training-loss comparison across activations.](../../artifacts/figures/activation_comparison/train_loss_comparison.png)

![Figure 5. Validation-loss comparison across activations.](../../artifacts/figures/activation_comparison/val_loss_comparison.png)

![Figure 6. Validation-accuracy comparison across activations.](../../artifacts/figures/activation_comparison/val_accuracy_comparison.png)

Under the short horizon, ReLU looked like the natural final choice. However, this ranking changed after longer optimization. A dedicated 200-epoch comparison between ReLU and Leaky ReLU showed that Leaky ReLU eventually overtook ReLU.

| Activation | Epochs | Best validation accuracy | Best epoch |
| --- | --- | --- | --- |
| ReLU | 200 | 0.9122 | 140 |
| Leaky ReLU | 200 | 0.9132 | 81 |

This long-run result is important because it shows that the short-run ranking was not final. Observation: Leaky ReLU achieves a slightly higher peak and reaches it earlier. Explanation: the longer horizon exposes a small but consistent optimization advantage that is not fully visible in the 20-epoch runs. Conclusion: Leaky ReLU is selected for the final model.

### 4.3 Weight-Decay Comparison

The weight-decay study was carried out with ReLU, hidden dimension 128, and learning rate 0.3 fixed. The results show that mild regularization helps, but the gain is modest. Zero regularization and 5e-4 are both close to the best value, while 1e-3 is clearly too strong.

| Weight decay | Best validation accuracy | Best epoch |
| --- | --- | --- |
| 0 | 0.9012 | 19 |
| 1e-4 | 0.9023 | 20 |
| 5e-4 | 0.9010 | 19 |
| 1e-3 | 0.8978 | 20 |

![Figure 7. Training-loss comparison across weight-decay values.](../../artifacts/figures/weight_decay_comparison/train_loss_comparison.png)

![Figure 8. Validation-loss comparison across weight-decay values.](../../artifacts/figures/weight_decay_comparison/val_loss_comparison.png)

![Figure 9. Validation-accuracy comparison across weight-decay values.](../../artifacts/figures/weight_decay_comparison/val_accuracy_comparison.png)

Observation: a small L2 penalty produces the strongest validation result, while stronger regularization degrades performance. Explanation: weak regularization suppresses excessive parameter growth without removing too much capacity. Conclusion: 1e-4 is retained in the final setting.

### 4.4 Hidden-Dimension Comparison

The hidden-dimension study examined whether additional capacity continued to improve validation accuracy. Performance increased steadily from 64 to 512, which justified moving away from the original 128-dimensional baseline. The extended comparison then checked whether the trend continued beyond 512. It did not. Both 768 and 1024 fell slightly below 512 while substantially increasing training cost.

| Hidden dimension | Best validation accuracy | Best epoch | Training time (s) |
| --- | --- | --- | --- |
| 64 | 0.8967 | 17 | 16.38 |
| 128 | 0.9023 | 20 | 27.05 |
| 256 | 0.9043 | 19 | 68.65 |
| 512 | 0.9100 | 20 | 154.74 |
| 768 | 0.9078 | 19 | 307.92 |
| 1024 | 0.9077 | 20 | 372.23 |

![Figure 10. Training-loss comparison across hidden dimensions.](../../artifacts/figures/hidden_dim_comparison/train_loss_comparison.png)

![Figure 11. Validation-loss comparison across hidden dimensions.](../../artifacts/figures/hidden_dim_comparison/val_loss_comparison.png)

![Figure 12. Validation-accuracy comparison across hidden dimensions.](../../artifacts/figures/hidden_dim_comparison/val_accuracy_comparison.png)

Observation: validation performance improves up to 512 and then saturates. Explanation: increasing width initially raises representational capacity, but the later gains become too small to offset optimization cost and mild overcapacity. Conclusion: 512 is the strongest width in the tested range.

### 4.5 Scheduler Comparison

Scheduler comparison was conducted under the strongest short-run setting with hidden dimension 512. Step decay produced the highest validation accuracy, with cosine decay only slightly behind. No decay was clearly weaker, and exponential decay fell much further behind.

| Scheduler | Best validation accuracy | Best epoch |
| --- | --- | --- |
| None | 0.9033 | 19 |
| Step | 0.9100 | 20 |
| Exponential | 0.8852 | 9 |
| Cosine | 0.9093 | 18 |

![Figure 13. Training-loss comparison across schedulers.](../../artifacts/figures/scheduler_comparison/train_loss_comparison.png)

![Figure 14. Validation-loss comparison across schedulers.](../../artifacts/figures/scheduler_comparison/val_loss_comparison.png)

![Figure 15. Validation-accuracy comparison across schedulers.](../../artifacts/figures/scheduler_comparison/val_accuracy_comparison.png)

Observation: step and cosine are both strong, but step remains slightly ahead in the 20-epoch comparison. Explanation: short-run staged decay keeps the learning rate high enough for fast progress while still reducing it in later epochs. Conclusion: step decay is kept for the final model, but its interval is lengthened for the 200-epoch run.

### 4.6 Final Model Curves

The final model uses Leaky ReLU, hidden dimension 512, learning rate 0.3, weight decay 1e-4, and a 200-epoch step schedule with decay every 50 epochs. The best validation result appears at epoch 81. This confirms that validation-based checkpoint selection is necessary even in a long training run.

![Figure 16. Final training and validation loss curves.](../../artifacts/runs/final_model/figures/loss_curves.png)

![Figure 17. Final validation accuracy curve.](../../artifacts/runs/final_model/figures/val_accuracy_curve.png)

The long-run curves also show a mild form of overfitting when loss and accuracy are examined together. Training loss continues to decrease throughout the run, but validation loss reaches its minimum much earlier and then fluctuates at a higher level. Validation accuracy behaves differently: it remains high and reaches its best value at epoch 81 before entering a relatively stable plateau. This combination indicates that prolonged optimization still improves training-set fit, but it no longer improves validation calibration in the same way. The effect is therefore better described as mild long-run overfitting rather than catastrophic degradation. Observation: train loss keeps decreasing, validation loss bottoms out early, and validation accuracy peaks later but remains stable. Explanation: the model becomes increasingly specialized to the training distribution after the point of best validation loss, while the classification boundary remains broadly effective. Conclusion: fixed-epoch training is acceptable, but best-checkpoint selection is necessary because the final epoch is not the best model.

### 4.7 Final Test Accuracy and Confusion Matrix

The final selected checkpoint achieved 0.9006 accuracy on the test set.

![Figure 18. Confusion matrix on the final test set.](../../artifacts/runs/final_model/figures/confusion_matrix.png)

Per-class performance is uneven. Trouser, sneaker, bag, and ankle boot are recognized with very high accuracy, while shirt remains the weakest class. The confusion matrix indicates that the dominant failure pattern is not random noise but overlap among visually similar upper-body categories.

## 5. Representation Analysis

### 5.1 First-Layer Weight Visualization

The first-layer weights, reshaped back into 28 by 28 maps, show structured spatial patterns rather than diffuse noise. Several neurons emphasize vertical central regions, lower-body contours, or broad edge transitions around clothing boundaries.

![Figure 19. First-layer weight visualization.](../../artifacts/runs/final_model/figures/first_layer_weights.png)

Observation: many first-layer neurons respond to broad garment structure. Explanation: even without convolutions, the model can still exploit coarse spatial regularities after flattening. Conclusion: the first layer learns meaningful region-level templates rather than purely unstructured global weights.

### 5.2 Error Analysis

Misclassified examples are concentrated in semantically similar classes. Shirt is frequently confused with T-shirt/top, pullover, or coat, and some dress-like and coat-like instances remain ambiguous at 28 by 28 resolution.

![Figure 20. Representative misclassified test samples.](../../artifacts/runs/final_model/figures/error_cases.png)

Observation: the errors cluster around visually plausible category boundaries. Explanation: flattening weakens explicit local spatial structure, which makes subtle region-level differences harder to preserve. Conclusion: the model captures broad class structure but remains limited on fine-grained upper-body distinctions.

### 5.3 Hidden-Feature Visualization

The t-SNE projection of the penultimate hidden representation provides a qualitative view of class separation. Several classes form compact clusters, especially those with distinctive silhouettes. Upper-body garment classes remain more entangled.

![Figure 21. t-SNE projection of hidden representations.](../../artifacts/runs/final_model/figures/tsne_hidden_repr.png)

Observation: some clusters are well separated while others overlap substantially. Explanation: strong silhouette classes are easier for the MLP to encode in a linearly separable form, whereas classes with weak contour contrast remain mixed. Conclusion: the learned representation is discriminative but not uniformly clean across all categories.

## 6. Conclusion

The final model was obtained through staged search rather than through a single exhaustive sweep. The main findings are consistent across the experiments. First, the initial learning-rate range was too small, and the useful region extended to 0.3. Second, mild weight decay outperformed both stronger regularization and none at all. Third, hidden-dimension gains saturated beyond 512, which made 512 the best width in the tested range. Fourth, step decay remained slightly stronger than cosine under the short-run comparison and provided a suitable backbone for long training. Finally, the activation ranking changed with training horizon: ReLU led in short runs, but Leaky ReLU slightly surpassed it in the 200-epoch comparison and was therefore selected for the final model.

The final checkpoint achieved 0.9132 validation accuracy and 0.9006 test accuracy. The confusion matrix, error cases, and t-SNE projection indicate that the main remaining weakness lies in distinguishing visually similar upper-body categories after flattening. Even with that limitation, the system satisfies the assignment requirements and provides a complete, reproducible experimental pipeline.

## 7. Repository and External Resources

- GitHub repository: To be filled
- Model-weight download link: To be filled
