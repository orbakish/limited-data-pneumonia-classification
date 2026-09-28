# Results analysis - both seeds

Written by `python -m src.analysis` from `results/`; change the code, not this file. Every difference
is second minus first, paired on the same test images, with a 95% interval from 2000 patient-group resamples
shared by every run. The intervals cover test-sampling noise for fixed trained models, not training
noise - the seed rows show that. No comparison is primary: the tables count how often a pattern holds
(NOTES.md, how the stability results are read); the report gives the two seeds' mean and calls a finding
strong only when both seeds agree (NOTES.md #32). All the numbers: `comparisons.csv` and `runs.csv`.

**Views.** Seed 42 and seed 43 are each seed's own paired difference. *Consistent*: both seeds agree in
sign (+ both above 0, - both below). *Strict*: consistent, and both intervals exclude 0. *Median mean*:
the median over the cells of the two seeds' mean difference.

## Checks against the results files

Largest absolute difference between what the analysis recomputes and what the results files hold:

- metrics: 0.0e+00
- stability: 0.0e+00
- seed-42 intervals: 0.0e+00
- validation metrics: 0.0e+00

## Augmentation regime, within architecture and data size (9 cells each)

| Comparison | Measure | Seed 42 > 0 | Seed 43 > 0 | Consistent + / - | Strict + / - | Median mean |
|---|---|---|---|---|---|---|
| mild-none | ROC-AUC | 7/9 | 8/9 | 6 / 0 | 2 / 0 | +0.0100 |
| mild-none | balanced accuracy @0.5 | 7/9 | 8/9 | 7 / 1 | 4 / 0 | +0.0372 |
| mild-none | balanced accuracy, tuned threshold | 8/9 | 7/9 | 6 / 0 | 4 / 0 | +0.0346 |
| mild-none | specificity @0.5 | 7/9 | 8/9 | 7 / 1 | 4 / 0 | +0.0769 |
| mild-none | recall @0.5 | 2/9 | 4/9 | 1 / 1 | 0 / 0 | +0.0000 |
| mild-none | stability in-family | 6/9 | 6/9 | 5 / 2 | 2 / 1 | +0.0312 |
| mild-none | stability out-of-family | 4/9 | 4/9 | 4 / 5 | 3 / 4 | -0.0696 |
| mild-none | stability gap | 7/9 | 6/9 | 5 / 1 | 4 / 0 | +0.0508 |
| mild-none | gap, images correct in both runs | 7/9 | 6/9 | 5 / 1 | 3 / 0 | +0.0526 |
| mild-none | gap, empty map = 0 | 7/9 | 6/9 | 5 / 1 | 5 / 0 | +0.0518 |
| mild-none | out-of-family, empty map = 0 | 3/9 | 4/9 | 3 / 5 | 3 / 4 | -0.0695 |
| mild-none | share of empty maps under noise | 4/9 | 3/9 | 2 / 0 | 1 / 0 | +0.0000 |
| strong-none | ROC-AUC | 9/9 | 9/9 | 9 / 0 | 7 / 0 | +0.0203 |
| strong-none | balanced accuracy @0.5 | 9/9 | 9/9 | 9 / 0 | 5 / 0 | +0.0406 |
| strong-none | balanced accuracy, tuned threshold | 9/9 | 6/9 | 6 / 0 | 4 / 0 | +0.0419 |
| strong-none | specificity @0.5 | 9/9 | 9/9 | 9 / 0 | 5 / 0 | +0.0876 |
| strong-none | recall @0.5 | 5/9 | 3/9 | 2 / 2 | 0 / 0 | +0.0013 |
| strong-none | stability in-family | 6/9 | 7/9 | 6 / 2 | 6 / 1 | +0.0656 |
| strong-none | stability out-of-family | 4/9 | 5/9 | 3 / 3 | 2 / 2 | -0.0036 |
| strong-none | stability gap | 8/9 | 6/9 | 5 / 0 | 5 / 0 | +0.0644 |
| strong-none | gap, images correct in both runs | 8/9 | 8/9 | 7 / 0 | 5 / 0 | +0.0818 |
| strong-none | gap, empty map = 0 | 8/9 | 6/9 | 5 / 0 | 5 / 0 | +0.0644 |
| strong-none | out-of-family, empty map = 0 | 4/9 | 4/9 | 3 / 4 | 2 / 2 | -0.0270 |
| strong-none | share of empty maps under noise | 4/9 | 2/9 | 2 / 2 | 1 / 1 | -0.0011 |
| strong-mild | ROC-AUC | 8/9 | 7/9 | 7 / 1 | 2 / 0 | +0.0090 |
| strong-mild | balanced accuracy @0.5 | 8/9 | 5/9 | 4 / 0 | 4 / 0 | +0.0118 |
| strong-mild | balanced accuracy, tuned threshold | 8/9 | 4/9 | 3 / 0 | 2 / 0 | +0.0173 |
| strong-mild | specificity @0.5 | 8/9 | 5/9 | 4 / 0 | 4 / 0 | +0.0235 |
| strong-mild | recall @0.5 | 4/9 | 3/9 | 1 / 1 | 0 / 0 | +0.0000 |
| strong-mild | stability in-family | 6/9 | 8/9 | 5 / 0 | 4 / 0 | +0.0118 |
| strong-mild | stability out-of-family | 5/9 | 4/9 | 2 / 2 | 1 / 1 | +0.0087 |
| strong-mild | stability gap | 3/9 | 5/9 | 2 / 3 | 1 / 1 | +0.0088 |
| strong-mild | gap, images correct in both runs | 4/9 | 6/9 | 3 / 2 | 1 / 0 | +0.0182 |
| strong-mild | gap, empty map = 0 | 3/9 | 5/9 | 2 / 3 | 1 / 1 | +0.0065 |
| strong-mild | out-of-family, empty map = 0 | 5/9 | 4/9 | 2 / 2 | 1 / 1 | +0.0089 |
| strong-mild | share of empty maps under noise | 4/9 | 3/9 | 2 / 1 | 0 / 0 | -0.0011 |

## Data size, within architecture and regime (3 cells each)

| Comparison | Measure | Seed 42 > 0 | Seed 43 > 0 | Consistent + / - | Strict + / - | Median mean |
|---|---|---|---|---|---|---|
| 50%-20%, None | ROC-AUC | 0/3 | 1/3 | 0 / 2 | 0 / 0 | -0.0064 |
| 50%-20%, None | balanced accuracy @0.5 | 0/3 | 0/3 | 0 / 3 | 0 / 2 | -0.0451 |
| 50%-20%, None | balanced accuracy, tuned threshold | 0/3 | 0/3 | 0 / 3 | 0 / 3 | -0.0699 |
| 50%-20%, None | specificity @0.5 | 0/3 | 0/3 | 0 / 3 | 0 / 2 | -0.0940 |
| 50%-20%, None | recall @0.5 | 1/3 | 2/3 | 1 / 1 | 0 / 0 | +0.0038 |
| 50%-20%, None | stability in-family | 2/3 | 1/3 | 1 / 1 | 1 / 1 | +0.0162 |
| 50%-20%, None | stability out-of-family | 1/3 | 1/3 | 0 / 1 | 0 / 0 | -0.0093 |
| 50%-20%, None | stability gap | 1/3 | 1/3 | 0 / 1 | 0 / 1 | +0.0372 |
| 50%-20%, None | gap, images correct in both runs | 2/3 | 1/3 | 1 / 1 | 0 / 0 | +0.0184 |
| 50%-20%, None | gap, empty map = 0 | 1/3 | 1/3 | 0 / 1 | 0 / 1 | +0.0392 |
| 50%-20%, None | out-of-family, empty map = 0 | 1/3 | 1/3 | 0 / 1 | 0 / 1 | -0.0115 |
| 50%-20%, None | share of empty maps under noise | 1/3 | 2/3 | 0 / 0 | 0 / 0 | +0.0064 |
| 50%-20%, Mild | ROC-AUC | 1/3 | 2/3 | 1 / 1 | 0 / 0 | -0.0006 |
| 50%-20%, Mild | balanced accuracy @0.5 | 0/3 | 1/3 | 0 / 2 | 0 / 2 | -0.0252 |
| 50%-20%, Mild | balanced accuracy, tuned threshold | 0/3 | 2/3 | 0 / 1 | 0 / 0 | -0.0169 |
| 50%-20%, Mild | specificity @0.5 | 0/3 | 1/3 | 0 / 2 | 0 / 2 | -0.0556 |
| 50%-20%, Mild | recall @0.5 | 2/3 | 3/3 | 2 / 0 | 0 / 0 | +0.0051 |
| 50%-20%, Mild | stability in-family | 3/3 | 2/3 | 2 / 0 | 1 / 0 | +0.0353 |
| 50%-20%, Mild | stability out-of-family | 2/3 | 1/3 | 0 / 0 | 0 / 0 | -0.0022 |
| 50%-20%, Mild | stability gap | 2/3 | 2/3 | 1 / 0 | 1 / 0 | +0.0341 |
| 50%-20%, Mild | gap, images correct in both runs | 1/3 | 2/3 | 1 / 1 | 1 / 0 | +0.0397 |
| 50%-20%, Mild | gap, empty map = 0 | 2/3 | 2/3 | 1 / 0 | 1 / 0 | +0.0345 |
| 50%-20%, Mild | out-of-family, empty map = 0 | 2/3 | 1/3 | 0 / 0 | 0 / 0 | +0.0003 |
| 50%-20%, Mild | share of empty maps under noise | 1/3 | 1/3 | 0 / 1 | 0 / 0 | -0.0053 |
| 50%-20%, Strong | ROC-AUC | 1/3 | 3/3 | 1 / 0 | 0 / 0 | +0.0027 |
| 50%-20%, Strong | balanced accuracy @0.5 | 0/3 | 0/3 | 0 / 3 | 0 / 0 | -0.0241 |
| 50%-20%, Strong | balanced accuracy, tuned threshold | 1/3 | 0/3 | 0 / 2 | 0 / 1 | -0.0274 |
| 50%-20%, Strong | specificity @0.5 | 0/3 | 0/3 | 0 / 3 | 0 / 0 | -0.0470 |
| 50%-20%, Strong | recall @0.5 | 2/3 | 2/3 | 2 / 0 | 1 / 0 | +0.0051 |
| 50%-20%, Strong | stability in-family | 3/3 | 2/3 | 2 / 0 | 1 / 0 | +0.0133 |
| 50%-20%, Strong | stability out-of-family | 3/3 | 1/3 | 1 / 0 | 1 / 0 | +0.0244 |
| 50%-20%, Strong | stability gap | 1/3 | 2/3 | 1 / 1 | 0 / 1 | -0.0111 |
| 50%-20%, Strong | gap, images correct in both runs | 1/3 | 2/3 | 1 / 1 | 1 / 1 | -0.0189 |
| 50%-20%, Strong | gap, empty map = 0 | 1/3 | 2/3 | 1 / 1 | 0 / 1 | -0.0104 |
| 50%-20%, Strong | out-of-family, empty map = 0 | 3/3 | 1/3 | 1 / 0 | 1 / 0 | +0.0264 |
| 50%-20%, Strong | share of empty maps under noise | 0/3 | 2/3 | 0 / 1 | 0 / 0 | -0.0021 |
| 100%-20%, None | ROC-AUC | 0/3 | 3/3 | 0 / 0 | 0 / 0 | -0.0036 |
| 100%-20%, None | balanced accuracy @0.5 | 0/3 | 1/3 | 0 / 2 | 0 / 2 | -0.0662 |
| 100%-20%, None | balanced accuracy, tuned threshold | 0/3 | 1/3 | 0 / 2 | 0 / 2 | -0.0776 |
| 100%-20%, None | specificity @0.5 | 0/3 | 1/3 | 0 / 2 | 0 / 2 | -0.1389 |
| 100%-20%, None | recall @0.5 | 2/3 | 2/3 | 2 / 1 | 0 / 0 | +0.0064 |
| 100%-20%, None | stability in-family | 0/3 | 0/3 | 0 / 3 | 0 / 2 | -0.0555 |
| 100%-20%, None | stability out-of-family | 1/3 | 2/3 | 1 / 1 | 0 / 1 | +0.0382 |
| 100%-20%, None | stability gap | 0/3 | 1/3 | 0 / 2 | 0 / 1 | -0.0471 |
| 100%-20%, None | gap, images correct in both runs | 2/3 | 1/3 | 1 / 1 | 1 / 1 | -0.0395 |
| 100%-20%, None | gap, empty map = 0 | 0/3 | 1/3 | 0 / 2 | 0 / 1 | -0.0472 |
| 100%-20%, None | out-of-family, empty map = 0 | 1/3 | 2/3 | 1 / 1 | 1 / 1 | +0.0422 |
| 100%-20%, None | share of empty maps under noise | 1/3 | 1/3 | 1 / 1 | 0 / 1 | +0.0000 |
| 100%-20%, Mild | ROC-AUC | 1/3 | 1/3 | 0 / 1 | 0 / 0 | -0.0051 |
| 100%-20%, Mild | balanced accuracy @0.5 | 0/3 | 1/3 | 0 / 2 | 0 / 1 | -0.0212 |
| 100%-20%, Mild | balanced accuracy, tuned threshold | 0/3 | 1/3 | 0 / 2 | 0 / 1 | -0.0363 |
| 100%-20%, Mild | specificity @0.5 | 0/3 | 1/3 | 0 / 2 | 0 / 1 | -0.0513 |
| 100%-20%, Mild | recall @0.5 | 3/3 | 2/3 | 2 / 0 | 0 / 0 | +0.0090 |
| 100%-20%, Mild | stability in-family | 3/3 | 2/3 | 2 / 0 | 1 / 0 | +0.0132 |
| 100%-20%, Mild | stability out-of-family | 3/3 | 2/3 | 2 / 0 | 0 / 0 | +0.0351 |
| 100%-20%, Mild | stability gap | 0/3 | 1/3 | 0 / 2 | 0 / 1 | -0.0219 |
| 100%-20%, Mild | gap, images correct in both runs | 0/3 | 1/3 | 0 / 2 | 0 / 0 | -0.0252 |
| 100%-20%, Mild | gap, empty map = 0 | 1/3 | 1/3 | 0 / 1 | 0 / 1 | -0.0221 |
| 100%-20%, Mild | out-of-family, empty map = 0 | 3/3 | 2/3 | 2 / 0 | 1 / 0 | +0.0399 |
| 100%-20%, Mild | share of empty maps under noise | 1/3 | 1/3 | 1 / 1 | 0 / 1 | -0.0043 |
| 100%-20%, Strong | ROC-AUC | 1/3 | 2/3 | 1 / 1 | 1 / 0 | +0.0001 |
| 100%-20%, Strong | balanced accuracy @0.5 | 0/3 | 1/3 | 0 / 2 | 0 / 0 | -0.0205 |
| 100%-20%, Strong | balanced accuracy, tuned threshold | 0/3 | 1/3 | 0 / 2 | 0 / 1 | -0.0077 |
| 100%-20%, Strong | specificity @0.5 | 0/3 | 1/3 | 0 / 2 | 0 / 1 | -0.0513 |
| 100%-20%, Strong | recall @0.5 | 3/3 | 1/3 | 1 / 0 | 0 / 0 | +0.0038 |
| 100%-20%, Strong | stability in-family | 3/3 | 2/3 | 2 / 0 | 1 / 0 | +0.0370 |
| 100%-20%, Strong | stability out-of-family | 2/3 | 1/3 | 1 / 1 | 1 / 0 | +0.0157 |
| 100%-20%, Strong | stability gap | 1/3 | 2/3 | 1 / 1 | 1 / 1 | -0.0217 |
| 100%-20%, Strong | gap, images correct in both runs | 1/3 | 2/3 | 1 / 1 | 1 / 1 | -0.0264 |
| 100%-20%, Strong | gap, empty map = 0 | 1/3 | 2/3 | 1 / 1 | 1 / 1 | -0.0207 |
| 100%-20%, Strong | out-of-family, empty map = 0 | 2/3 | 1/3 | 1 / 1 | 1 / 0 | +0.0162 |
| 100%-20%, Strong | share of empty maps under noise | 1/3 | 1/3 | 1 / 0 | 1 / 0 | -0.0011 |

## Chance-adjusted stability: does rescaling change the sign? (regime, 9 cells each)

| Comparison | Seed | In-family agrees | Out-of-family agrees | Gap agrees |
|---|---|---|---|---|
| mild-none | 42 | 6/9 | 6/9 | 9/9 |
| mild-none | 43 | 6/9 | 7/9 | 9/9 |
| strong-none | 42 | 6/9 | 4/9 | 9/9 |
| strong-none | 43 | 7/9 | 5/9 | 9/9 |
| strong-mild | 42 | 4/9 | 5/9 | 9/9 |
| strong-mild | 43 | 8/9 | 8/9 | 9/9 |

## Seed yardstick: seed 43 minus seed 42, per experiment (27)

| Measure | Median abs | 90th percentile abs | Largest abs | Intervals excluding 0 |
|---|---|---|---|---|
| ROC-AUC | 0.0056 | 0.0226 | 0.0376 | 11/27 |
| balanced accuracy @0.5 | 0.0252 | 0.0592 | 0.0769 | 16/27 |
| balanced accuracy, tuned threshold | 0.0231 | 0.0676 | 0.1201 | 14/27 |
| specificity @0.5 | 0.0556 | 0.1205 | 0.1538 | 16/27 |
| recall @0.5 | 0.0026 | 0.0097 | 0.0154 | 3/27 |
| stability in-family | 0.0323 | 0.0813 | 0.1003 | 19/27 |
| stability out-of-family | 0.0401 | 0.1272 | 0.2103 | 21/27 |
| stability gap | 0.0470 | 0.0912 | 0.1780 | 23/27 |
| gap, images correct in both runs | 0.0499 | 0.1055 | 0.1500 | 21/27 |
| gap, empty map = 0 | 0.0476 | 0.0952 | 0.1828 | 21/27 |
| out-of-family, empty map = 0 | 0.0404 | 0.1310 | 0.2171 | 21/27 |
| share of empty maps under noise | 0.0021 | 0.0859 | 0.1154 | 7/27 |

## Tuned threshold minus 0.5, per experiment (27)

| Metric | Seed | Better | Worse | Same | Mean difference |
|---|---|---|---|---|---|
| balanced accuracy | 42 | 5 | 15 | 7 | -0.0077 |
| balanced accuracy | 43 | 10 | 11 | 6 | +0.0012 |
| specificity | 42 | 4 | 15 | 8 | -0.0163 |
| specificity | 43 | 10 | 11 | 6 | +0.0022 |
| recall | 42 | 6 | 2 | 19 | +0.0009 |
| recall | 43 | 8 | 8 | 11 | +0.0001 |
| f1 | 42 | 5 | 15 | 7 | -0.0035 |
| f1 | 43 | 10 | 11 | 6 | +0.0006 |

## Errors and prediction flips, summed over the 27 experiments

| Seed | False alarms @0.5 | Misses @0.5 | False alarms, tuned | Misses, tuned | Rotation flips to PNEUMONIA / to NORMAL | Noise flips to PNEUMONIA / to NORMAL |
|---|---|---|---|---|---|---|
| 42 | 2267 | 90 | 2370 | 80 | 1228 / 719 | 5272 / 28 |
| 43 | 2169 | 100 | 2155 | 99 | 1143 / 726 | 5269 / 40 |

## Empty maps under noise, per architecture (of 468 subset images)

| Architecture | Seed | Fewest | Most | Runs above 10% |
|---|---|---|---|---|
| MobileNetV3-Small | 42 | 0 | 2 | 0/9 |
| MobileNetV3-Small | 43 | 0 | 1 | 0/9 |
| ResNet18 | 42 | 6 | 52 | 2/9 |
| ResNet18 | 43 | 21 | 93 | 5/9 |
| EfficientNet-B0 | 42 | 0 | 5 | 0/9 |
| EfficientNet-B0 | 43 | 0 | 6 | 0/9 |

## Data size and the operating point (NOTES.md #33; 9 cells each)

The validation rows and the NORMAL images' margins are point estimates, without intervals: validation is
the set the checkpoint and the threshold were chosen on, so its values are optimistic, and paired
validation intervals would need a second cache. The margin is the PNEUMONIA logit minus the NORMAL logit;
a model says PNEUMONIA above 0 (the 0.5 threshold), so NORMAL margins rising towards 0 mean more false
alarms at any fixed threshold. Specificity counts NORMAL images only, so the class balance of the two
sets (validation 202 / 583, test 234 / 390) cannot explain a change in it.

| Comparison | Measure | Seed 42 up / equal / down | Seed 43 up / equal / down | Consistent + / - | Strict + / - | Median mean |
|---|---|---|---|---|---|---|
| 50%-20% | test specificity @0.5 | 0 / 0 / 9 | 1 / 0 / 8 | 0 / 8 | 0 / 4 | -0.0556 |
| 50%-20% | test specificity, tuned threshold | 1 / 0 / 8 | 2 / 0 / 7 | 0 / 6 | 0 / 3 | -0.0769 |
| 50%-20% | test recall @0.5 | 5 / 2 / 2 | 7 / 0 / 2 | 5 / 1 | 1 / 0 | +0.0051 |
| 50%-20% | test ROC-AUC | 2 / 0 / 7 | 6 / 0 / 3 | 2 / 3 | 0 / 0 | -0.0014 |
| 50%-20% | validation specificity @0.5 | 8 / 1 / 0 | 7 / 2 / 0 | 6 / 0 | - | +0.0124 |
| 50%-20% | validation ROC-AUC | 8 / 0 / 1 | 8 / 0 / 1 | 7 / 0 | - | +0.0017 |
| 50%-20% | NORMAL images' median margin, validation | 1 / 0 / 8 | 5 / 0 / 4 | 1 / 4 | - | -0.3756 |
| 50%-20% | NORMAL images' median margin, test | 6 / 0 / 3 | 8 / 0 / 1 | 5 / 0 | - | +0.9387 |
| 100%-20% | test specificity @0.5 | 0 / 0 / 9 | 3 / 0 / 6 | 0 / 6 | 0 / 4 | -0.0684 |
| 100%-20% | test specificity, tuned threshold | 0 / 0 / 9 | 3 / 0 / 6 | 0 / 6 | 0 / 4 | -0.0791 |
| 100%-20% | test recall @0.5 | 8 / 0 / 1 | 5 / 1 / 3 | 5 / 1 | 0 / 0 | +0.0064 |
| 100%-20% | test ROC-AUC | 2 / 0 / 7 | 6 / 0 / 3 | 1 / 2 | 1 / 0 | -0.0036 |
| 100%-20% | validation specificity @0.5 | 8 / 1 / 0 | 8 / 1 / 0 | 8 / 0 | - | +0.0149 |
| 100%-20% | validation ROC-AUC | 9 / 0 / 0 | 8 / 0 / 1 | 8 / 0 | - | +0.0025 |
| 100%-20% | NORMAL images' median margin, validation | 4 / 0 / 5 | 4 / 0 / 5 | 3 / 4 | - | -0.3658 |
| 100%-20% | NORMAL images' median margin, test | 8 / 0 / 1 | 6 / 0 / 3 | 5 / 0 | - | +0.8559 |

Epochs trained, more data against less (up / equal / down of 18): seed 42 11 / 3 / 4; seed 43 4 / 3 / 11.

**Diagnostic only, never a result** - the threshold is chosen on the test set itself: balanced accuracy there is 0.871-0.942 over the 54 runs, against 0.715-0.877 at 0.5.

## Architecture levels: larger minus smaller, within data size and regime (3 cells each)

Classification only: two architectures start from different stability gaps (NOTES.md, the reading rule).

| Comparison | Measure | Seed 42 up / equal / down | Seed 43 up / equal / down | Consistent + / - | Strict + / - | Median mean |
|---|---|---|---|---|---|---|
| EfficientNet-B0 minus MobileNetV3-Small, 20% | ROC-AUC | 1 / 0 / 2 | 1 / 0 / 2 | 1 / 2 | 0 / 0 | -0.0061 |
| EfficientNet-B0 minus MobileNetV3-Small, 20% | balanced accuracy, tuned threshold | 1 / 0 / 2 | 1 / 0 / 2 | 0 / 1 | 0 / 0 | -0.0047 |
| EfficientNet-B0 minus MobileNetV3-Small, 20% | specificity @0.5 | 0 / 0 / 3 | 0 / 0 / 3 | 0 / 3 | 0 / 0 | -0.0192 |
| EfficientNet-B0 minus MobileNetV3-Small, 20% | recall @0.5 | 1 / 2 / 0 | 3 / 0 / 0 | 1 / 0 | 1 / 0 | +0.0064 |
| EfficientNet-B0 minus MobileNetV3-Small, 50% | ROC-AUC | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 0 | -0.0053 |
| EfficientNet-B0 minus MobileNetV3-Small, 50% | balanced accuracy, tuned threshold | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 0 | -0.0096 |
| EfficientNet-B0 minus MobileNetV3-Small, 50% | specificity @0.5 | 2 / 0 / 1 | 2 / 0 / 1 | 2 / 1 | 0 / 1 | +0.0171 |
| EfficientNet-B0 minus MobileNetV3-Small, 50% | recall @0.5 | 2 / 0 / 1 | 2 / 0 / 1 | 1 / 0 | 0 / 0 | +0.0038 |
| EfficientNet-B0 minus MobileNetV3-Small, 100% | ROC-AUC | 2 / 0 / 1 | 2 / 0 / 1 | 2 / 1 | 1 / 0 | +0.0095 |
| EfficientNet-B0 minus MobileNetV3-Small, 100% | balanced accuracy, tuned threshold | 1 / 0 / 2 | 1 / 0 / 2 | 0 / 1 | 0 / 0 | +0.0038 |
| EfficientNet-B0 minus MobileNetV3-Small, 100% | specificity @0.5 | 1 / 0 / 2 | 3 / 0 / 0 | 1 / 0 | 1 / 0 | +0.0021 |
| EfficientNet-B0 minus MobileNetV3-Small, 100% | recall @0.5 | 2 / 1 / 0 | 1 / 2 / 0 | 1 / 0 | 0 / 0 | +0.0013 |
| ResNet18 minus MobileNetV3-Small, 20% | ROC-AUC | 2 / 0 / 1 | 2 / 0 / 1 | 2 / 1 | 0 / 0 | +0.0039 |
| ResNet18 minus MobileNetV3-Small, 20% | balanced accuracy, tuned threshold | 1 / 0 / 2 | 0 / 0 / 3 | 0 / 2 | 0 / 0 | -0.0248 |
| ResNet18 minus MobileNetV3-Small, 20% | specificity @0.5 | 2 / 0 / 1 | 0 / 0 / 3 | 0 / 1 | 0 / 1 | -0.0321 |
| ResNet18 minus MobileNetV3-Small, 20% | recall @0.5 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 1 / 0 | +0.0128 |
| ResNet18 minus MobileNetV3-Small, 50% | ROC-AUC | 0 / 0 / 3 | 3 / 0 / 0 | 0 / 0 | 0 / 0 | -0.0006 |
| ResNet18 minus MobileNetV3-Small, 50% | balanced accuracy, tuned threshold | 0 / 0 / 3 | 0 / 0 / 3 | 0 / 3 | 0 / 0 | -0.0389 |
| ResNet18 minus MobileNetV3-Small, 50% | specificity @0.5 | 0 / 0 / 3 | 0 / 0 / 3 | 0 / 3 | 0 / 1 | -0.0983 |
| ResNet18 minus MobileNetV3-Small, 50% | recall @0.5 | 3 / 0 / 0 | 2 / 1 / 0 | 2 / 0 | 0 / 0 | +0.0077 |
| ResNet18 minus MobileNetV3-Small, 100% | ROC-AUC | 1 / 0 / 2 | 3 / 0 / 0 | 1 / 0 | 0 / 0 | -0.0041 |
| ResNet18 minus MobileNetV3-Small, 100% | balanced accuracy, tuned threshold | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 1 | 0 / 1 | +0.0058 |
| ResNet18 minus MobileNetV3-Small, 100% | specificity @0.5 | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 1 | 0 / 0 | +0.0043 |
| ResNet18 minus MobileNetV3-Small, 100% | recall @0.5 | 1 / 2 / 0 | 2 / 0 / 1 | 1 / 0 | 0 / 0 | +0.0026 |
| ResNet18 minus EfficientNet-B0, 20% | ROC-AUC | 2 / 0 / 1 | 2 / 0 / 1 | 2 / 1 | 0 / 0 | +0.0100 |
| ResNet18 minus EfficientNet-B0, 20% | balanced accuracy, tuned threshold | 1 / 0 / 2 | 1 / 0 / 2 | 1 / 2 | 0 / 0 | -0.0201 |
| ResNet18 minus EfficientNet-B0, 20% | specificity @0.5 | 2 / 0 / 1 | 1 / 0 / 2 | 1 / 1 | 0 / 1 | -0.0128 |
| ResNet18 minus EfficientNet-B0, 20% | recall @0.5 | 2 / 0 / 1 | 2 / 0 / 1 | 2 / 1 | 1 / 0 | +0.0064 |
| ResNet18 minus EfficientNet-B0, 50% | ROC-AUC | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 1 | 1 / 0 | +0.0047 |
| ResNet18 minus EfficientNet-B0, 50% | balanced accuracy, tuned threshold | 1 / 0 / 2 | 2 / 0 / 1 | 0 / 0 | 0 / 0 | -0.0410 |
| ResNet18 minus EfficientNet-B0, 50% | specificity @0.5 | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 2 | -0.1132 |
| ResNet18 minus EfficientNet-B0, 50% | recall @0.5 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 0 / 0 | +0.0038 |
| ResNet18 minus EfficientNet-B0, 100% | ROC-AUC | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 1 | 0 / 0 | -0.0089 |
| ResNet18 minus EfficientNet-B0, 100% | balanced accuracy, tuned threshold | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 1 | 1 / 1 | +0.0019 |
| ResNet18 minus EfficientNet-B0, 100% | specificity @0.5 | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 1 | 0 / 1 | -0.0556 |
| ResNet18 minus EfficientNet-B0, 100% | recall @0.5 | 0 / 2 / 1 | 1 / 1 / 1 | 0 / 0 | 0 / 0 | +0.0000 |

## Interactions: the larger model's effect minus the smaller model's (3 cells each)

Positive: the larger model gains more from augmentation, or from more data, than the smaller one - the
proposal's question whether lightweight and larger models behave differently under limited data.

| Comparison | Measure | Seed 42 up / equal / down | Seed 43 up / equal / down | Consistent + / - | Strict + / - | Median mean |
|---|---|---|---|---|---|---|
| mild-none, EfficientNet-B0 minus MobileNetV3-Small | ROC-AUC | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 2 / 0 | +0.0184 |
| mild-none, EfficientNet-B0 minus MobileNetV3-Small | balanced accuracy, tuned threshold | 2 / 0 / 1 | 1 / 0 / 2 | 0 / 0 | 0 / 0 | +0.0053 |
| mild-none, EfficientNet-B0 minus MobileNetV3-Small | specificity @0.5 | 2 / 0 / 1 | 2 / 0 / 1 | 1 / 0 | 0 / 0 | +0.0000 |
| mild-none, EfficientNet-B0 minus MobileNetV3-Small | stability gap | 1 / 0 / 2 | 1 / 0 / 2 | 0 / 1 | 0 / 1 | -0.0486 |
| strong-none, EfficientNet-B0 minus MobileNetV3-Small | ROC-AUC | 3 / 0 / 0 | 2 / 0 / 1 | 2 / 0 | 1 / 0 | +0.0113 |
| strong-none, EfficientNet-B0 minus MobileNetV3-Small | balanced accuracy, tuned threshold | 2 / 0 / 1 | 1 / 0 / 2 | 1 / 1 | 0 / 0 | +0.0105 |
| strong-none, EfficientNet-B0 minus MobileNetV3-Small | specificity @0.5 | 2 / 0 / 1 | 0 / 0 / 3 | 0 / 1 | 0 / 0 | -0.0214 |
| strong-none, EfficientNet-B0 minus MobileNetV3-Small | stability gap | 1 / 0 / 2 | 1 / 0 / 2 | 0 / 1 | 0 / 1 | -0.0588 |
| 50%-20%, EfficientNet-B0 minus MobileNetV3-Small | ROC-AUC | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 0 | -0.0055 |
| 50%-20%, EfficientNet-B0 minus MobileNetV3-Small | balanced accuracy, tuned threshold | 1 / 0 / 2 | 1 / 0 / 2 | 0 / 1 | 0 / 0 | -0.0049 |
| 50%-20%, EfficientNet-B0 minus MobileNetV3-Small | specificity @0.5 | 2 / 0 / 1 | 2 / 0 / 1 | 2 / 1 | 0 / 0 | +0.0363 |
| 50%-20%, EfficientNet-B0 minus MobileNetV3-Small | stability gap | 1 / 0 / 2 | 3 / 0 / 0 | 1 / 0 | 1 / 0 | +0.0502 |
| 100%-20%, EfficientNet-B0 minus MobileNetV3-Small | ROC-AUC | 2 / 0 / 1 | 2 / 0 / 1 | 1 / 0 | 1 / 0 | +0.0022 |
| 100%-20%, EfficientNet-B0 minus MobileNetV3-Small | balanced accuracy, tuned threshold | 1 / 0 / 2 | 2 / 0 / 1 | 0 / 0 | 0 / 0 | +0.0064 |
| 100%-20%, EfficientNet-B0 minus MobileNetV3-Small | specificity @0.5 | 1 / 0 / 2 | 3 / 0 / 0 | 1 / 0 | 1 / 0 | +0.0427 |
| 100%-20%, EfficientNet-B0 minus MobileNetV3-Small | stability gap | 1 / 0 / 2 | 2 / 0 / 1 | 0 / 0 | 0 / 0 | +0.0036 |
| mild-none, ResNet18 minus MobileNetV3-Small | ROC-AUC | 1 / 0 / 2 | 1 / 0 / 2 | 1 / 2 | 0 / 0 | -0.0091 |
| mild-none, ResNet18 minus MobileNetV3-Small | balanced accuracy, tuned threshold | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 1 | 0 / 0 | -0.0107 |
| mild-none, ResNet18 minus MobileNetV3-Small | specificity @0.5 | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 1 | -0.0556 |
| mild-none, ResNet18 minus MobileNetV3-Small | stability gap | 1 / 0 / 2 | 0 / 0 / 3 | 0 / 2 | 0 / 2 | -0.1070 |
| strong-none, ResNet18 minus MobileNetV3-Small | ROC-AUC | 1 / 0 / 2 | 0 / 0 / 3 | 0 / 2 | 0 / 0 | -0.0024 |
| strong-none, ResNet18 minus MobileNetV3-Small | balanced accuracy, tuned threshold | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 1 | 1 / 0 | -0.0006 |
| strong-none, ResNet18 minus MobileNetV3-Small | specificity @0.5 | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 1 | 0 / 1 | -0.0128 |
| strong-none, ResNet18 minus MobileNetV3-Small | stability gap | 1 / 0 / 2 | 1 / 0 / 2 | 1 / 2 | 1 / 2 | -0.0928 |
| 50%-20%, ResNet18 minus MobileNetV3-Small | ROC-AUC | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 1 | 0 / 0 | -0.0045 |
| 50%-20%, ResNet18 minus MobileNetV3-Small | balanced accuracy, tuned threshold | 1 / 0 / 2 | 2 / 0 / 1 | 0 / 0 | 0 / 0 | -0.0259 |
| 50%-20%, ResNet18 minus MobileNetV3-Small | specificity @0.5 | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 1 | 0 / 0 | -0.0577 |
| 50%-20%, ResNet18 minus MobileNetV3-Small | stability gap | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 3 / 0 | +0.0877 |
| 100%-20%, ResNet18 minus MobileNetV3-Small | ROC-AUC | 1 / 0 / 2 | 3 / 0 / 0 | 1 / 0 | 0 / 0 | +0.0009 |
| 100%-20%, ResNet18 minus MobileNetV3-Small | balanced accuracy, tuned threshold | 2 / 0 / 1 | 2 / 0 / 1 | 2 / 1 | 0 / 0 | +0.0306 |
| 100%-20%, ResNet18 minus MobileNetV3-Small | specificity @0.5 | 1 / 0 / 2 | 3 / 0 / 0 | 1 / 0 | 0 / 0 | +0.0919 |
| 100%-20%, ResNet18 minus MobileNetV3-Small | stability gap | 2 / 0 / 1 | 1 / 0 / 2 | 1 / 1 | 1 / 1 | +0.0002 |
| mild-none, ResNet18 minus EfficientNet-B0 | ROC-AUC | 0 / 0 / 3 | 0 / 0 / 3 | 0 / 3 | 0 / 2 | -0.0270 |
| mild-none, ResNet18 minus EfficientNet-B0 | balanced accuracy, tuned threshold | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 1 | 0 / 0 | +0.0026 |
| mild-none, ResNet18 minus EfficientNet-B0 | specificity @0.5 | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 1 | -0.1111 |
| mild-none, ResNet18 minus EfficientNet-B0 | stability gap | 1 / 0 / 2 | 2 / 0 / 1 | 0 / 0 | 0 / 0 | -0.0585 |
| strong-none, ResNet18 minus EfficientNet-B0 | ROC-AUC | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 0 | -0.0137 |
| strong-none, ResNet18 minus EfficientNet-B0 | balanced accuracy, tuned threshold | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 1 | 0 / 1 | +0.0276 |
| strong-none, ResNet18 minus EfficientNet-B0 | specificity @0.5 | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 1 | 0 / 1 | +0.0299 |
| strong-none, ResNet18 minus EfficientNet-B0 | stability gap | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 1 | 0 / 0 | -0.0017 |
| 50%-20%, ResNet18 minus EfficientNet-B0 | ROC-AUC | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 1 | 0 / 0 | +0.0006 |
| 50%-20%, ResNet18 minus EfficientNet-B0 | balanced accuracy, tuned threshold | 1 / 0 / 2 | 2 / 0 / 1 | 0 / 0 | 0 / 0 | -0.0182 |
| 50%-20%, ResNet18 minus EfficientNet-B0 | specificity @0.5 | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 1 | 0 / 1 | -0.0491 |
| 50%-20%, ResNet18 minus EfficientNet-B0 | stability gap | 3 / 0 / 0 | 1 / 0 / 2 | 1 / 0 | 0 / 0 | +0.0034 |
| 100%-20%, ResNet18 minus EfficientNet-B0 | ROC-AUC | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 1 | 0 / 1 | -0.0108 |
| 100%-20%, ResNet18 minus EfficientNet-B0 | balanced accuracy, tuned threshold | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 1 | 1 / 1 | +0.0220 |
| 100%-20%, ResNet18 minus EfficientNet-B0 | specificity @0.5 | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 1 | 0 / 1 | +0.0684 |
| 100%-20%, ResNet18 minus EfficientNet-B0 | stability gap | 2 / 0 / 1 | 2 / 0 / 1 | 1 / 0 | 1 / 0 | -0.0316 |

## Per architecture: what augmentation and more data do (3 cells each)

| Comparison | Measure | Seed 42 up / equal / down | Seed 43 up / equal / down | Consistent + / - | Strict + / - | Median mean |
|---|---|---|---|---|---|---|
| MobileNetV3-Small, mild-none | ROC-AUC | 3 / 0 / 0 | 2 / 0 / 1 | 2 / 0 | 0 / 0 | +0.0100 |
| MobileNetV3-Small, mild-none | balanced accuracy, tuned threshold | 3 / 0 / 0 | 2 / 0 / 1 | 2 / 0 | 2 / 0 | +0.0346 |
| MobileNetV3-Small, mild-none | specificity @0.5 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 2 / 0 | +0.0791 |
| MobileNetV3-Small, mild-none | stability gap | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 3 / 0 | +0.0874 |
| MobileNetV3-Small, strong-none | ROC-AUC | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 2 / 0 | +0.0162 |
| MobileNetV3-Small, strong-none | balanced accuracy, tuned threshold | 3 / 0 / 0 | 2 / 0 / 1 | 2 / 0 | 2 / 0 | +0.0737 |
| MobileNetV3-Small, strong-none | specificity @0.5 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 3 / 0 | +0.1560 |
| MobileNetV3-Small, strong-none | stability gap | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 3 / 0 | +0.0886 |
| MobileNetV3-Small, 50%-20% | ROC-AUC | 2 / 0 / 1 | 2 / 0 / 1 | 2 / 1 | 0 / 0 | +0.0016 |
| MobileNetV3-Small, 50%-20% | balanced accuracy, tuned threshold | 1 / 0 / 2 | 1 / 0 / 2 | 0 / 1 | 0 / 1 | -0.0120 |
| MobileNetV3-Small, 50%-20% | specificity @0.5 | 0 / 0 / 3 | 0 / 0 / 3 | 0 / 3 | 0 / 2 | -0.0919 |
| MobileNetV3-Small, 50%-20% | stability gap | 1 / 0 / 2 | 0 / 0 / 3 | 0 / 2 | 0 / 2 | -0.0470 |
| MobileNetV3-Small, 100%-20% | ROC-AUC | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 0 | -0.0032 |
| MobileNetV3-Small, 100%-20% | balanced accuracy, tuned threshold | 0 / 0 / 3 | 0 / 0 / 3 | 0 / 3 | 0 / 2 | -0.0449 |
| MobileNetV3-Small, 100%-20% | specificity @0.5 | 0 / 0 / 3 | 0 / 0 / 3 | 0 / 3 | 0 / 3 | -0.1303 |
| MobileNetV3-Small, 100%-20% | stability gap | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 2 | -0.0221 |
| ResNet18, mild-none | ROC-AUC | 1 / 0 / 2 | 3 / 0 / 0 | 1 / 0 | 0 / 0 | +0.0041 |
| ResNet18, mild-none | balanced accuracy, tuned threshold | 2 / 0 / 1 | 3 / 0 / 0 | 2 / 0 | 0 / 0 | +0.0239 |
| ResNet18, mild-none | specificity @0.5 | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 1 | 0 / 0 | +0.0513 |
| ResNet18, mild-none | stability gap | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 1 | 0 / 0 | -0.0196 |
| ResNet18, strong-none | ROC-AUC | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 2 / 0 | +0.0179 |
| ResNet18, strong-none | balanced accuracy, tuned threshold | 3 / 0 / 0 | 2 / 0 / 1 | 2 / 0 | 1 / 0 | +0.0419 |
| ResNet18, strong-none | specificity @0.5 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 1 / 0 | +0.0876 |
| ResNet18, strong-none | stability gap | 2 / 0 / 1 | 2 / 0 / 1 | 1 / 0 | 1 / 0 | +0.0010 |
| ResNet18, 50%-20% | ROC-AUC | 0 / 0 / 3 | 3 / 0 / 0 | 0 / 0 | 0 / 0 | -0.0018 |
| ResNet18, 50%-20% | balanced accuracy, tuned threshold | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 1 | -0.0380 |
| ResNet18, 50%-20% | specificity @0.5 | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 1 | -0.0962 |
| ResNet18, 50%-20% | stability gap | 3 / 0 / 0 | 2 / 0 / 1 | 2 / 0 | 1 / 0 | +0.0406 |
| ResNet18, 100%-20% | ROC-AUC | 0 / 0 / 3 | 3 / 0 / 0 | 0 / 0 | 0 / 0 | -0.0051 |
| ResNet18, 100%-20% | balanced accuracy, tuned threshold | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 1 | 0 / 1 | -0.0256 |
| ResNet18, 100%-20% | specificity @0.5 | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 1 | 0 / 0 | -0.0684 |
| ResNet18, 100%-20% | stability gap | 1 / 0 / 2 | 1 / 0 / 2 | 1 / 2 | 1 / 1 | -0.0219 |
| EfficientNet-B0, mild-none | ROC-AUC | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 2 / 0 | +0.0285 |
| EfficientNet-B0, mild-none | balanced accuracy, tuned threshold | 3 / 0 / 0 | 2 / 0 / 1 | 2 / 0 | 2 / 0 | +0.0449 |
| EfficientNet-B0, mild-none | specificity @0.5 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 2 / 0 | +0.0769 |
| EfficientNet-B0, mild-none | stability gap | 3 / 0 / 0 | 1 / 0 / 2 | 1 / 0 | 1 / 0 | +0.0388 |
| EfficientNet-B0, strong-none | ROC-AUC | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 3 / 0 | +0.0317 |
| EfficientNet-B0, strong-none | balanced accuracy, tuned threshold | 3 / 0 / 0 | 2 / 0 / 1 | 2 / 0 | 1 / 0 | +0.0415 |
| EfficientNet-B0, strong-none | specificity @0.5 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 | 1 / 0 | +0.0620 |
| EfficientNet-B0, strong-none | stability gap | 3 / 0 / 0 | 1 / 0 / 2 | 1 / 0 | 1 / 0 | +0.0446 |
| EfficientNet-B0, 50%-20% | ROC-AUC | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 0 | -0.0044 |
| EfficientNet-B0, 50%-20% | balanced accuracy, tuned threshold | 0 / 0 / 3 | 0 / 0 / 3 | 0 / 3 | 0 / 2 | -0.0274 |
| EfficientNet-B0, 50%-20% | specificity @0.5 | 0 / 0 / 3 | 0 / 0 / 3 | 0 / 3 | 0 / 1 | -0.0513 |
| EfficientNet-B0, 50%-20% | stability gap | 0 / 0 / 3 | 3 / 0 / 0 | 0 / 0 | 0 / 0 | +0.0372 |
| EfficientNet-B0, 100%-20% | ROC-AUC | 2 / 0 / 1 | 2 / 0 / 1 | 1 / 0 | 1 / 0 | -0.0036 |
| EfficientNet-B0, 100%-20% | balanced accuracy, tuned threshold | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 1 | -0.0363 |
| EfficientNet-B0, 100%-20% | specificity @0.5 | 0 / 0 / 3 | 1 / 0 / 2 | 0 / 2 | 0 / 1 | -0.0513 |
| EfficientNet-B0, 100%-20% | stability gap | 0 / 0 / 3 | 2 / 0 / 1 | 0 / 1 | 0 / 0 | -0.0217 |
