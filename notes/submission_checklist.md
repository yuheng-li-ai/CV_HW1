# Submission Checklist

## Final Configuration

- [ ] Confirm that the final configuration in `configs/final_model.yaml` is the intended submission version.
- [ ] Confirm that the final checkpoint path is correct and accessible.
- [ ] Confirm that the reported validation and test metrics match the final checkpoint.

## Code and Repository

- [ ] Verify that the repository is public.
- [ ] Verify that `README.md` is fully written in English.
- [ ] Verify that `README.md` contains environment setup instructions.
- [ ] Verify that `README.md` contains training, evaluation, and visualization commands.
- [ ] Verify that the repository does not include forbidden frameworks such as PyTorch, TensorFlow, or JAX in the training pipeline.
- [ ] Verify that all required modules are present: data loading, model definition, training loop, evaluation, and hyperparameter search.
- [ ] Remove accidental temporary files if they are not needed for submission.

## Final Results

- [ ] Confirm the final validation accuracy.
- [ ] Confirm the final test accuracy.
- [ ] Confirm the final best epoch.
- [ ] Confirm that the final confusion matrix is generated.
- [ ] Confirm that the first-layer weight visualization is generated.
- [ ] Confirm that the misclassification figure is generated.
- [ ] Confirm that the hidden-feature t-SNE figure is generated.

## Report

- [ ] Verify that the report is fully written in English.
- [ ] Verify that the writing style is formal and paper-like.
- [ ] Verify that the abstract, introduction, method, experiments, results, analysis, and conclusion are all present.
- [ ] Verify that the report explicitly states that the implementation is from scratch with NumPy.
- [ ] Verify that the report explains the validation-based checkpoint-selection policy.
- [ ] Verify that the report includes the training-loss curve.
- [ ] Verify that the report includes the validation-loss curve.
- [ ] Verify that the report includes the validation-accuracy curve.
- [ ] Verify that the report includes the confusion matrix.
- [ ] Verify that the report includes the first-layer weight visualization.
- [ ] Verify that the report includes the error-analysis figure.
- [ ] Verify that the report includes the t-SNE figure if you decide to keep it.
- [ ] Verify that the report discusses why the final configuration was selected.
- [ ] Verify that the report discusses the main confusion patterns and representative failure cases.

## External Links

- [ ] Add the final GitHub repository link to the report.
- [ ] Upload the final trained weights to Google Drive or another external host.
- [ ] Add the final weight-download link to the report.
- [ ] Verify that both links are accessible without permission issues.

## Final Verification

- [ ] Re-run the final evaluation command once before submission if time permits.
- [ ] Re-check that the final checkpoint and figures are still present after the last run.
- [ ] Verify that all dates and timestamps required by the assignment are earlier than the deadline.
- [ ] Export the final report PDF and inspect it visually before submission.
