# Results Organization

This directory stores the outputs of all experiments.

Each training run should save enough information to identify the experiment and compare it with the other configurations.

## Required Outputs Per Experiment

For every experiment (E01–E27), save:

* Experiment ID
* Model architecture
* Training-data size
* Augmentation regime
* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Grad-CAM stability score
* Training notes if relevant

## Metrics

Numerical metrics should be saved under:

```text
results/metrics/
```

Recommended naming format:

```text
E01_metrics.json
E02_metrics.json
...
E27_metrics.json
```

Each metrics file should contain at least:

```text
experiment_id
model
data_size
augmentation
accuracy
precision
recall
f1
gradcam_stability
```

## Figures

Plots and visual results should be saved under:

```text
results/figures/
```

Recommended filenames:

```text
E01_confusion_matrix.png
E01_gradcam_examples.png
E01_training_curves.png
```

The same naming convention should be used for all experiments.

## Final Comparison

After all experiments are completed, the main results should also be summarized in one combined table for comparison across:

* Model architecture
* Training-data size
* Augmentation strength

This table will be used for the final analysis, report, and presentation.

## Important

Use the experiment IDs defined in:

```text
experiments/experiment_log.md
```

Do not overwrite results from previous experiments.
