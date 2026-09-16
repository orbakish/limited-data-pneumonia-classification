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
* Recall (on PNEUMONIA) and specificity (on NORMAL)
* F1-score
* Balanced accuracy and ROC-AUC
* A 95% confidence interval for each of those
* Confusion matrix
* Grad-CAM stability: one score per perturbation family (in-family and out-of-family)
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

Each metrics file is written by `src/evaluate.py` and contains (schema agreed in NOTES.md
question #13; every headline metric also has `_ci_low` and `_ci_high`, omitted here for
brevity except on `roc_auc`):

```json
{
  "experiment_id": "E01",
  "model": "MobileNetV3-Small",
  "data_size": "20%",
  "augmentation": "None",
  "seed": 42,
  "fine_tuning": "full",
  "weighted_loss": true,
  "images": 624,
  "patient_groups": 428,
  "roc_auc": 0.981,
  "balanced_accuracy": 0.939,
  "accuracy": 0.944,
  "precision": 0.958,
  "recall": 0.951,
  "specificity": 0.927,
  "f1": 0.954,
  "roc_auc_ci_low": 0.97,
  "roc_auc_ci_high": 0.99,
  "confusion_matrix": {
    "tn": 217,
    "fp": 17,
    "fn": 19,
    "tp": 371
  },
  "gradcam_stability_in_family": null,
  "gradcam_stability_out_of_family": null,
  "bootstrap": {
    "samples": 2000,
    "skipped": 0,
    "seed": 42,
    "unit": "patient group",
    "confidence": 95
  }
}
```

Notes on the fields:

* **`recall` is on PNEUMONIA and `specificity` on NORMAL**, so both error rates are reportable
  as numbers rather than only through the confusion-matrix image.
* **The confidence intervals are a percentile bootstrap over patient groups**, not over
  individual images: one patient can contribute several test images, and resampling images as
  if they were independent would make the intervals too narrow. `bootstrap` records the
  settings so any interval can be regenerated.
* **The interval covers test-sampling noise only** - how much the number would move on another
  test set of the same size. Run-to-run training noise is a separate question (NOTES.md #21).
* **The two stability fields stay `null`** until `src/stability.py` exists; the gap between
  them is derived rather than stored.
* Labels (`model`, `data_size`, `augmentation`) use the same wording as
  `experiments/experiment_log.md`, so the two files can be read side by side.

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

The same naming convention should be used for all experiments - `src/figures.py` writes them
with exactly these names, three per experiment (81 in total).

Four further figures summarise the whole sweep and are the ones the report uses:

```text
summary_roc_auc.png          test ROC-AUC vs data size, one panel per architecture
summary_f1.png               the same at the 0.5 threshold
summary_stability.png        Grad-CAM stability under noise (out-of-family)
summary_stability_gap.png    in-family minus out-of-family: how much invariance is trained in
```

In every figure, colour means the augmentation regime - blue `none`, vermillion `mild`, green
`strong` - and each regime also has its own marker, so the figures stay readable in greyscale
and for colour-blind readers. Each PNG stores its own title inside the file, and
`python -m src.figures --audit` checks that all 81 per-experiment figures carry the label their
config implies, which is how we catch a mislabelled or stale figure before it reaches the
report.

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
