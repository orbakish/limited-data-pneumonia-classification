# Results analysis - both seeds

Written by `python -m src.analysis` from `results/`; change the code, not this file. Every difference
is second minus first, paired on the same test images, with a 95% interval from 2000 patient-group resamples
shared by every run. The intervals cover test-sampling noise for fixed trained models, not training
noise - the seed rows show that. No comparison is primary: the tables count how often a pattern holds
(NOTES.md, how the stability results are read), and which seed view the report uses is open
(NOTES.md #32). All the numbers: `comparisons.csv` and `runs.csv`.

**Views.** Seed 42 and seed 43 are each seed's own paired difference. *Consistent*: both seeds agree in
sign (+ both above 0, - both below). *Strict*: consistent, and both intervals exclude 0. *Median mean*:
the median over the cells of the two seeds' mean difference.

## Checks against the results files

Largest absolute difference between what the analysis recomputes and what the results files hold:

- metrics: 0.0e+00
- stability: 0.0e+00
- seed-42 intervals: 0.0e+00

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
