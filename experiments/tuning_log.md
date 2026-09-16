# Tuning Log

Nine validation-only runs that fix the training recipe before the 27 experiments start. Kept
separate from [experiment_log.md](experiment_log.md) so validation numbers are never mixed with
test results. Columns as agreed in NOTES.md question #22.

**Constant in all nine runs** (stated once rather than repeated per row): training subset
`train_20` (889 images), augmentation `none`, seed 42, AdamW with weight decay 1e-4, learning
rate 1e-3 for the new head and 1e-4 for the backbone, batch size 32, at most 30 epochs with
early stopping after 5 without improvement. **Every number below comes from our own validation
split (785 images); the official test split is untouched until the sweep.**

**Checkpoint selection and early stopping follow validation balanced accuracy** (changed from
ROC-AUC on 2026-09-16, after the first T01 run measured the problem): validation ROC-AUC
saturated at 0.995-0.996 from epoch 5 onwards, a spread smaller than its own 95% interval
(+-0.0025), while balanced accuracy was still climbing from 0.92 to 0.95. Selecting on ROC-AUC
kept the epoch-7 checkpoint over epoch 10's, giving up 3.2 points of balanced accuracy and half
the validation loss to win by 0.0006 on a saturated metric. ROC-AUC is still reported everywhere.

What the three blocks decide:

* **T01-T03** - the baseline, one per architecture: full fine-tuning, class-weighted loss.
* **T04-T06** - partial fine-tuning against that baseline (NOTES.md question #6). Judged on
  validation ROC-AUC, with balanced accuracy as the tiebreak.
* **T07-T09** - no class weighting against that baseline (NOTES.md question #17). Judged on
  **balanced accuracy**, because weighting mostly moves the decision threshold and ROC-AUC is
  largely blind to that; ROC-AUC is the secondary check.

The default wins unless a variant is clearly better: with one seed per run, a gap of about 0.01
is only one or two standard errors, so a difference that size prompts a look at the training
curves rather than an automatic switch.

| T-ID | Model | Fine-tuning | Weighted loss | Best epoch | Val F1 | Val balanced accuracy | Val ROC-AUC (95% CI) | Selected | Notes |
| ---- | ----------------- | ------- | --- | ---: | ---: | ---: | ---: | --- | ----- |
| T01 | MobileNetV3-Small | full    | yes | TBD | TBD | TBD | TBD | TBD | baseline |
| T02 | ResNet18          | full    | yes | TBD | TBD | TBD | TBD | TBD | baseline |
| T03 | EfficientNet-B0   | full    | yes | TBD | TBD | TBD | TBD | TBD | baseline |
| T04 | MobileNetV3-Small | partial | yes | TBD | TBD | TBD | TBD | TBD | question #6 |
| T05 | ResNet18          | partial | yes | TBD | TBD | TBD | TBD | TBD | question #6 |
| T06 | EfficientNet-B0   | partial | yes | TBD | TBD | TBD | TBD | TBD | question #6 |
| T07 | MobileNetV3-Small | full    | no  | TBD | TBD | TBD | TBD | TBD | question #17 |
| T08 | ResNet18          | full    | no  | TBD | TBD | TBD | TBD | TBD | question #17 |
| T09 | EfficientNet-B0   | full    | no  | TBD | TBD | TBD | TBD | TBD | question #17 |

**Partial fine-tuning** means every block at the final 7x7 resolution plus the classifier:
`layer4` + `fc` in ResNet18, `features.9`-`features.12` + `classifier` in MobileNetV3-Small,
`features.6`-`features.8` + `classifier` in EfficientNet-B0. Frozen layers stay in `.eval()`
mode so their BatchNorm statistics do not drift.

## How to run them

```bash
python run_all.py --tuning            # all nine, skipping any that already finished
python run_experiment.py --id T01     # or one at a time
```

Each run prints its best epoch, validation ROC-AUC, balanced accuracy and F1 when it finishes,
and writes `checkpoints/<ID>_seed42/history.json` with a row per epoch plus
`results/figures/<ID>_training_curves.png`. The numbers in the table above are copied from that
line; the confidence interval comes from the bootstrap described in NOTES.md.

## Outcome

Filled in once the nine runs are done: which fine-tuning depth and which loss the 27
experiments use, and the reasoning. The protocol is then frozen, and the report describes this
phase as preliminary tuning.

| Decision | Chosen | Evidence |
| --- | --- | --- |
| Fine-tuning depth (#6) | TBD | TBD |
| Class weighting (#17) | TBD | TBD |
| Epoch budget still 30 with patience 5? | TBD | TBD |
