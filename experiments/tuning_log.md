# Tuning Log

Sixteen validation-only runs, plus extra seeds for six of them, that fixed the training recipe
before the 27 experiments start. Kept separate from [experiment_log.md](experiment_log.md) so
validation numbers are never mixed with test results. Columns as agreed in NOTES.md question #22.

**Constant in every run** (stated once rather than repeated per row): training subset
`train_20` (889 images), seed 42 unless the seed table says otherwise, AdamW with weight decay
1e-4, learning rate 1e-3 for the new head and 1e-4 for the backbone, batch size 32, at most 30
epochs with early stopping after 5 without improvement, a Tesla T4 with torch 2.11.0+cu128.
Augmentation is `none` except in T10-T15, where it is `strong`; the Notes column says so for
those rows. **Every number below comes from our own validation split (785 images); the official
test split is untouched until the sweep.**

**Code version.** T01-T09 and their extra seeds ran on the code at `main @ 2920210` (T01's
first run, on 2026-09-16, came from an earlier commit; a same-seed rerun at `2920210` was
bit-identical). T10-T16 and T10's second seed ran at `yotam/freeze-recipe @ 7345d76`, which
added T10-T16 without changing how T01-T09 train.

**Checkpoint selection and early stopping follow validation balanced accuracy** (changed from
ROC-AUC on 2026-09-16, after the first T01 run measured the problem): validation ROC-AUC
saturated at 0.995-0.996 from epoch 5 onwards, a spread smaller than its own 95% interval
(+-0.0025), while balanced accuracy was still climbing from 0.92 to 0.95. Selecting on ROC-AUC
kept the epoch-7 checkpoint over epoch 10's, giving up 3.2 points of balanced accuracy and half
the validation loss to win by 0.0006 on a saturated metric. ROC-AUC is still reported everywhere.

What each block decides or checks:

* **T01-T03** - the baseline, one per architecture: full fine-tuning, class-weighted loss.
* **T04-T06** - partial fine-tuning against that baseline (NOTES.md question #6). Judged on
  **balanced accuracy**, with ROC-AUC secondary: T01's ROC-AUC saturated inside its own
  interval, so a ROC-AUC gap here would almost always be a tie (changed 2026-09-25).
* **T07-T09** - no class weighting against that baseline (NOTES.md question #17). Judged on
  **balanced accuracy**, because weighting mostly moves the decision threshold and ROC-AUC is
  largely blind to that; ROC-AUC is the secondary check.
* **T10-T15** - the spot-check that question #6's caveat asked for: full fine-tuning (T10-T12)
  against partial (T13-T15) under strong augmentation, one run per architecture. **Rule stated
  before the runs:** partial stays unless full beats it by more than 0.01 in balanced accuracy
  on at least two of the three architectures; MobileNetV3-Small counts only if both T10 seeds
  (42 and 43) beat T13.
* **T16** - a diagnostic for NOTES.md question #26, **not a recipe candidate**: T01 with a
  BatchNorm momentum of 0.1 instead of MobileNetV3-Small's 0.01.

The default wins unless a variant is clearly better: with one seed per run, a gap of about 0.01
is only one or two standard errors, so a difference that size prompts a look at the training
curves rather than an automatic switch.

`Selected` is `yes` for the runs whose configuration became the frozen recipe in the comparison
that decided it (T04-T06), `no` for the alternatives they beat, and `-` for T10-T16, which
checked the decision rather than made it.

| T-ID | Model | Fine-tuning | Weighted loss | Best epoch | Val F1 | Val balanced accuracy | Val ROC-AUC (95% CI) | Selected | Notes |
| ---- | ----------------- | ------- | --- | ---: | ---: | ---: | ---: | --- | ----- |
| T01 | MobileNetV3-Small | full    | yes | 10 | 0.9770 | 0.9493 | 0.9954 (0.9924-0.9978) | no  | baseline; 15 epochs, stopped by patience, 8.2 min |
| T02 | ResNet18          | full    | yes | 6  | 0.9854 | 0.9741 | 0.9974 (0.9949-0.9991) | no  | baseline; 11 epochs, 6.4 min |
| T03 | EfficientNet-B0   | full    | yes | 6  | 0.9854 | 0.9741 | 0.9971 (0.9950-0.9988) | no  | baseline; 11 epochs, 6.4 min |
| T04 | MobileNetV3-Small | partial | yes | 4  | 0.9811 | 0.9666 | 0.9944 (0.9903-0.9975) | yes | question #6; 9 epochs, 4.9 min |
| T05 | ResNet18          | partial | yes | 2  | 0.9879 | 0.9815 | 0.9987 (0.9973-0.9996) | yes | question #6; 7 epochs, 4.0 min |
| T06 | EfficientNet-B0   | partial | yes | 9  | 0.9844 | 0.9781 | 0.9969 (0.9942-0.9989) | yes | question #6; 14 epochs, 8.0 min |
| T07 | MobileNetV3-Small | full    | no  | 10 | 0.9744 | 0.9468 | 0.9953 (0.9922-0.9976) | no  | question #17; 15 epochs, 8.4 min |
| T08 | ResNet18          | full    | no  | 11 | 0.9854 | 0.9757 | 0.9978 (0.9956-0.9994) | no  | question #17; 16 epochs, 9.5 min |
| T09 | EfficientNet-B0   | full    | no  | 4  | 0.9837 | 0.9675 | 0.9966 (0.9941-0.9985) | no  | question #17; 9 epochs, 5.1 min |
| T10 | MobileNetV3-Small | full    | yes | 18 | 0.9854 | 0.9725 | 0.9967 (0.9939-0.9988) | -   | spot-check, strong augmentation; 23 epochs, 13.6 min |
| T11 | ResNet18          | full    | yes | 6  | 0.9746 | 0.9719 | 0.9968 (0.9939-0.9989) | -   | spot-check, strong augmentation; 11 epochs, 7.1 min |
| T12 | EfficientNet-B0   | full    | yes | 6  | 0.9730 | 0.9637 | 0.9934 (0.9872-0.9979) | -   | spot-check, strong augmentation; 11 epochs, 6.9 min |
| T13 | MobileNetV3-Small | partial | yes | 15 | 0.9853 | 0.9806 | 0.9971 (0.9941-0.9992) | -   | spot-check, strong augmentation; 20 epochs, 11.7 min |
| T14 | ResNet18          | partial | yes | 2  | 0.9801 | 0.9706 | 0.9944 (0.9895-0.9980) | -   | spot-check, strong augmentation; 7 epochs, 4.3 min |
| T15 | EfficientNet-B0   | partial | yes | 10 | 0.9730 | 0.9637 | 0.9946 (0.9911-0.9975) | -   | spot-check, strong augmentation; 15 epochs, 9.1 min |
| T16 | MobileNetV3-Small | full    | yes | 7  | 0.9828 | 0.9715 | 0.9966 (0.9940-0.9985) | -   | question #26 diagnostic, BatchNorm momentum 0.1; 12 epochs, 6.6 min |

Every run stopped by patience, none by the 30-epoch cap.

**Partial fine-tuning** means every block at the final 7x7 resolution plus the classifier:
`layer4` + `fc` in ResNet18, `features.9`-`features.12` + `classifier` in MobileNetV3-Small,
`features.6`-`features.8` + `classifier` in EfficientNet-B0. Frozen layers stay in `.eval()`
mode so their BatchNorm statistics do not drift.

### Extra seeds

Run to measure how much a result moves when only the seed changes (NOTES.md question #21).
Same configuration as the row above with the same T-ID.

| T-ID | Seed | Best epoch | Val F1 | Val balanced accuracy | Val ROC-AUC (95% CI) | Notes |
| ---- | ---: | ---: | ---: | ---: | ---: | ----- |
| T01 | 43 | 11 | 0.9713 | 0.9571 | 0.9921 (0.9874-0.9959) | 16 epochs, 8.9 min |
| T01 | 44 | 6  | 0.9707 | 0.9182 | 0.9942 (0.9906-0.9972) | 11 epochs, 6.1 min |
| T02 | 43 | 4  | 0.9810 | 0.9714 | 0.9972 (0.9950-0.9987) | 9 epochs, 5.3 min |
| T03 | 43 | 6  | 0.9791 | 0.9762 | 0.9968 (0.9934-0.9990) | 11 epochs, 6.4 min |
| T04 | 43 | 11 | 0.9819 | 0.9674 | 0.9949 (0.9912-0.9980) | 16 epochs, 8.7 min |
| T04 | 44 | 4  | 0.9828 | 0.9699 | 0.9942 (0.9894-0.9979) | 9 epochs, 4.9 min |
| T09 | 43 | 6  | 0.9827 | 0.9731 | 0.9968 (0.9937-0.9989) | 11 epochs, 6.4 min |
| T10 | 43 | 18 | 0.9835 | 0.9805 | 0.9974 (0.9949-0.9990) | strong augmentation; 23 epochs, 13.6 min |

## How to run them

```bash
python run_all.py --tuning                   # all sixteen, skipping any that already finished
python run_experiment.py --id T01            # or one at a time
python run_experiment.py --id T01 --seed 43  # an extra seed
```

Each run prints its best epoch, validation ROC-AUC, balanced accuracy and F1 when it finishes,
and writes `checkpoints/<ID>_seed<N>/history.json` with a row per epoch,
`checkpoints/<ID>_seed<N>/validation_metrics.json` with the full metrics, and
`results/figures/<ID>[_seed<N>]_training_curves.png` (the seed appears in the figure name only
for seeds other than 42). The numbers in the tables above are copied from that line; the
confidence interval comes from the bootstrap described in NOTES.md.

## Outcome

Decided on 2026-09-26. The protocol is now frozen for E01-E27, and the report describes this
phase as preliminary tuning.

| Decision | Chosen | Evidence |
| --- | --- | --- |
| Fine-tuning depth (#6) | **Partial**, for all three architectures | MobileNetV3-Small: partial 0.9666 / 0.9674 / 0.9699 over three seeds against full 0.9493 / 0.9571 / 0.9182. T16 then showed that gap comes from BatchNorm running statistics, not from worse weights: with momentum 0.1, full reaches 0.9715 (NOTES.md #26). ResNet18 and EfficientNet-B0: partial ahead by 0.0074 and 0.0040, single seed, within noise. Applied to all three because the protocol fixes one recipe. **Spot-check under strong augmentation**, full minus partial: -0.0081 and -0.0001 (MobileNetV3-Small, seeds 42 and 43), +0.0013 (ResNet18), 0.0000 (EfficientNet-B0) - no full win on any architecture, so partial stays. |
| Class weighting (#17) | **Weighted** (kept) | Unweighted minus weighted: -0.0025 / +0.0016 / -0.0066 (T07-T09 against T01-T03), and -0.0031 for EfficientNet-B0 on seed 43. At the selected epoch only 1-3 validation images change. No detectable difference, so the default wins. |
| Epoch budget still 30 with patience 5? | **Yes**, verified at 20% data only | Every run stopped by patience; the latest stop was after epoch 23 (T10, strong augmentation, both seeds, best epoch 18), and no unaugmented run went past epoch 16. Augmented runs run longer. Not tested at 50% or 100% data, where an epoch has about 2.5x and 5x as many updates. |
