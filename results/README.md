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
* The threshold metrics at 0.5 and again at a threshold chosen on validation (NOTES.md #27)
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

A second seed's files carry the seed in their names - `E01_seed43_metrics.json` for seed 43 (NOTES.md #21) -
so they never overwrite seed 42's.

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
  "fine_tuning": "partial",
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
  "gradcam_stability_in_family": 0.91,
  "gradcam_stability_out_of_family": 0.84,
  "gradcam_stability_gap": 0.07,
  "stability": {
    "similarity": "pearson",
    "target": "true class",
    "subset": "data/splits/stability_subset.txt",
    "images": 468,
    "rotation_degrees": 5.0,
    "noise_sigma": 0.02,
    "...": "flip rates, chance baseline, cosine and correct-only means"
  },
  "bootstrap": {
    "samples": 2000,
    "skipped": 0,
    "seed": 42,
    "unit": "patient group",
    "confidence": 95
  },
  "threshold_logit_margin": 0.84,
  "threshold": 0.698,
  "threshold_rule": "max validation balanced accuracy on the selected checkpoint",
  "balanced_accuracy_tuned": 0.940,
  "accuracy_tuned": 0.938,
  "precision_tuned": 0.968,
  "recall_tuned": 0.931,
  "specificity_tuned": 0.949,
  "f1_tuned": 0.949,
  "confusion_matrix_tuned": {
    "tn": 222,
    "fp": 12,
    "fn": 27,
    "tp": 363
  }
}
```

Notes on the fields:

* **`recall` is on PNEUMONIA and `specificity` on NORMAL**, so both error rates are reportable
  as numbers rather than only through the confusion-matrix image.
* **Two decision thresholds.** The unsuffixed threshold metrics (balanced accuracy, accuracy,
  precision, recall, specificity, F1) and `confusion_matrix` are at 0.5. The `*_tuned` fields
  are the same metrics at a threshold chosen on the validation set - the one that maximises
  validation balanced accuracy for this run's checkpoint - and applied unchanged to test
  (NOTES.md #27). `threshold_logit_margin` is the value applied; `threshold` is its
  PNEUMONIA probability, for reading. ROC-AUC uses no threshold, so it has no tuned version.
  The tuned metrics have intervals too (`_tuned_ci_low` / `_tuned_ci_high`), from the same
  bootstrap with the threshold held fixed.
* **The confidence intervals are a percentile bootstrap over patient groups**, not over
  individual images: one patient can contribute several test images, and resampling images as
  if they were independent would make the intervals too narrow. `bootstrap` records the
  settings so any interval can be regenerated.
* **The interval covers test-sampling noise only** - how much the number would move on another
  test set of the same size. Run-to-run training noise is a separate question (NOTES.md #21).
* **Grad-CAM stability** (NOTES.md #7 and the stability decision): each map explains the
  image's true class, on a fixed subset of 468 test images; the number is the **Pearson
  correlation** between the map of the original image and the map of the perturbed one -
  in-family a +/-5 degree rotation (re-aligned before comparing), out-of-family Gaussian noise.
  `gradcam_stability_gap` is in-family minus out-of-family, over the same images; all three have
  intervals. The `stability` block records the settings and diagnostics. Tuning runs leave the
  two stability fields `null`.
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
with exactly these names, three per experiment (81 in total). The confusion-matrix figure shows
two panels side by side: at 0.5 and at the validation-chosen threshold.

Four further figures summarise the whole sweep and are the ones the report uses:

```text
summary_roc_auc.png          test ROC-AUC vs data size, one panel per architecture
summary_f1.png               test F1 at the validation-chosen threshold (at 0.5 if any run lacks one)
summary_stability.png        Grad-CAM stability under noise (out-of-family)
summary_stability_gap.png    in-family minus out-of-family: how much invariance is trained in
```

Seed 43's figures follow the same rule: `E01_seed43_confusion_matrix.png` and so on per experiment, and
`summary_roc_auc_seed43.png` and so on for the summaries. `python -m src.figures --report --seed 43` draws them.

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

`python -m src.figures --report` writes it as `results/summary_table.md` (seed 42) and
`results/summary_table_seed43.md` (seed 43), one line per experiment.

## Per-run files

```text
results/runs/E01_seed42/ ... results/runs/E27_seed43/
```

Gathered on Colab by `python -m src.analysis --collect --seed N` (54 folders, one per experiment and seed), so
the comparisons can be recomputed from the repo alone, without the checkpoints on Drive:

* `scores.json` - every validation and test image's two logits and label, from the run's best checkpoint scored
  again. `gate` records the check against the results file: all 54 runs are `PASS-exact`.
* `stability_per_image.json` - per subset image: class, prediction and every perturbed pass's Pearson score.
* `stability_maps.npy` - the raw 7x7 Grad-CAM maps, 468 images x 5 (clean + 4 perturbed).
* `validation_metrics.json` and `history.json` - the run's validation pass and its per-epoch training history.

## Analysis

```text
results/analysis/
```

Written by `python -m src.analysis` from the files above:

* `summary.md` - the tables to read first: the checks against the results files, then how often each pattern
  holds across the architecture and data-size cells, in each seed and in both.
* `comparisons.csv` - every comparison (augmentation regime, data size, tuned threshold against 0.5, seed 43
  against seed 42) as a paired difference with its 95% interval, one row per metric and seed view.
* `runs.csv` - one row per run: errors, stability, empty maps, and prediction flips by direction.
* `analysis_regime_roc_auc.png`, `analysis_regime_gap.png`, `analysis_flips.png` - the three figures.
* `bootstrap_cache.npz` - every run scored on the 2,000 shared resamples; it saves about 20 minutes when the
  analysis runs again, and is rebuilt automatically if anything it depends on changes.

## Important

Use the experiment IDs defined in:

```text
experiments/experiment_log.md
```

Do not overwrite results from previous experiments.
