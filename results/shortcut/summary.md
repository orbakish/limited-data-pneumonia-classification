# Shortcut check (NOTES.md #15) - covariates and regions

Written by `python -m src.shortcut` from the dataset and `results/runs/`; change the code, not this file.
The readings were stated in the plan before any model output was related to an image property (NOTES.md
#15). The intervention (Colab) and the audit are added in later steps.

## Checks

- Covariates of 4447 training, 785 validation and 624 test images.
- The module's preprocessing, normalised, against the evaluation transform: largest difference 0.0e+00 on 16 test images.
- Empty true-class maps, all runs: 205 of 25272.

## What the data offers: each covariate alone as a classifier (ROC-AUC)

Between the classes, higher values meaning PNEUMONIA (0.5 is no separation, below 0.5 the covariate is
higher for NORMAL); between healthy images, higher values meaning a test image rather than a training or
validation one (the shift of NOTES.md #33).

| Covariate | Classes, training pool | Classes, test | Healthy, test against training and validation |
|---|---|---|---|
| JPEG quantisation-table mean | 0.999 | 0.451 | 1.000 |
| file bytes per pixel | 0.001 | 0.444 | 0.002 |
| original megapixels | 0.092 | 0.061 | 0.528 |
| original aspect ratio (w/h) | 0.870 | 0.704 | 0.700 |
| mean intensity at 224 | 0.515 | 0.427 | 0.531 |
| contrast (std) at 224 | 0.297 | 0.338 | 0.453 |
| dark fraction at 224 | 0.395 | 0.442 | 0.509 |
| grain at 224 | 0.087 | 0.127 | 0.363 |
| grain at 224, horizontal | 0.180 | 0.177 | 0.505 |
| grain at 224, vertical | 0.055 | 0.117 | 0.245 |

## Do the margins follow them? Within one class, over the runs of each seed

Spearman correlation of each covariate with the logit margin (PNEUMONIA minus NORMAL), alone and given
the other seven (partial): the median over the 27 runs of a seed, and in brackets the lowest and highest.
The runs rank the same images, so they are replicates of one estimate, not votes. A covariate is named
only if its median partial |Spearman| is at least 0.20 in both seeds and larger than every
other covariate's (NOTES.md #15). Grain's horizontal and vertical parts are shown alone, as a check.

**Test, NORMAL images** - named: original megapixels

| Covariate | Spearman, seed 42 | Spearman, seed 43 | Partial, seed 42 | Partial, seed 43 |
|---|---|---|---|---|
| JPEG quantisation-table mean | the same for every image | | | |
| file bytes per pixel | +0.125 (-0.03 to +0.25) | +0.122 (+0.01 to +0.32) | +0.031 (-0.11 to +0.17) | +0.053 (-0.09 to +0.20) |
| original megapixels | -0.501 (-0.68 to -0.38) | -0.497 (-0.67 to -0.22) | -0.303 (-0.48 to -0.21) | -0.322 (-0.47 to -0.15) |
| original aspect ratio (w/h) | +0.408 (+0.29 to +0.58) | +0.411 (+0.17 to +0.57) | +0.096 (+0.02 to +0.22) | +0.077 (+0.01 to +0.15) |
| mean intensity at 224 | +0.101 (-0.05 to +0.20) | +0.116 (-0.08 to +0.26) | +0.173 (+0.07 to +0.37) | +0.180 (+0.07 to +0.28) |
| contrast (std) at 224 | -0.091 (-0.16 to +0.00) | -0.082 (-0.18 to +0.01) | -0.013 (-0.16 to +0.07) | -0.018 (-0.15 to +0.06) |
| dark fraction at 224 | -0.110 (-0.16 to +0.04) | -0.074 (-0.22 to +0.05) | +0.068 (-0.03 to +0.24) | +0.070 (+0.00 to +0.21) |
| grain at 224 | -0.214 (-0.30 to -0.09) | -0.177 (-0.36 to -0.07) | -0.110 (-0.25 to -0.04) | -0.103 (-0.25 to -0.01) |
| grain at 224, horizontal | -0.116 (-0.20 to +0.01) | -0.086 (-0.25 to +0.02) | - | - |
| grain at 224, vertical | -0.291 (-0.42 to -0.18) | -0.270 (-0.40 to -0.17) | - | - |

**Test, PNEUMONIA images** - named: none

| Covariate | Spearman, seed 42 | Spearman, seed 43 | Partial, seed 42 | Partial, seed 43 |
|---|---|---|---|---|
| JPEG quantisation-table mean | +0.127 (-0.20 to +0.28) | +0.102 (-0.11 to +0.31) | -0.029 (-0.22 to +0.09) | -0.031 (-0.19 to +0.05) |
| file bytes per pixel | -0.169 (-0.37 to +0.15) | -0.196 (-0.38 to +0.03) | -0.024 (-0.19 to +0.13) | -0.053 (-0.21 to +0.04) |
| original megapixels | -0.322 (-0.58 to -0.07) | -0.347 (-0.61 to -0.06) | -0.145 (-0.36 to +0.05) | -0.179 (-0.41 to -0.06) |
| original aspect ratio (w/h) | +0.248 (+0.07 to +0.42) | +0.266 (+0.11 to +0.44) | +0.124 (+0.05 to +0.27) | +0.118 (+0.01 to +0.27) |
| mean intensity at 224 | -0.005 (-0.23 to +0.16) | -0.005 (-0.25 to +0.13) | +0.011 (-0.08 to +0.14) | +0.044 (-0.08 to +0.11) |
| contrast (std) at 224 | -0.121 (-0.36 to +0.14) | -0.167 (-0.34 to +0.12) | +0.020 (-0.12 to +0.13) | -0.003 (-0.13 to +0.09) |
| dark fraction at 224 | -0.048 (-0.25 to +0.14) | -0.119 (-0.32 to +0.13) | +0.006 (-0.10 to +0.11) | +0.003 (-0.05 to +0.07) |
| grain at 224 | -0.265 (-0.47 to -0.05) | -0.325 (-0.45 to +0.01) | -0.061 (-0.24 to +0.08) | -0.069 (-0.25 to +0.11) |
| grain at 224, horizontal | -0.112 (-0.32 to +0.04) | -0.169 (-0.30 to +0.07) | - | - |
| grain at 224, vertical | -0.381 (-0.58 to -0.10) | -0.424 (-0.58 to -0.04) | - | - |

**Validation, NORMAL images** - named: original aspect ratio (w/h)

| Covariate | Spearman, seed 42 | Spearman, seed 43 | Partial, seed 42 | Partial, seed 43 |
|---|---|---|---|---|
| JPEG quantisation-table mean | +0.241 (+0.18 to +0.30) | +0.241 (+0.18 to +0.29) | +0.235 (+0.16 to +0.33) | +0.245 (+0.18 to +0.33) |
| file bytes per pixel | -0.004 (-0.15 to +0.09) | -0.023 (-0.18 to +0.11) | +0.076 (-0.03 to +0.21) | +0.067 (-0.05 to +0.28) |
| original megapixels | -0.231 (-0.43 to -0.06) | -0.271 (-0.50 to -0.04) | +0.033 (-0.16 to +0.18) | -0.007 (-0.22 to +0.16) |
| original aspect ratio (w/h) | +0.346 (+0.22 to +0.52) | +0.311 (+0.12 to +0.48) | +0.294 (+0.19 to +0.43) | +0.274 (+0.03 to +0.41) |
| mean intensity at 224 | +0.044 (-0.06 to +0.18) | +0.056 (-0.10 to +0.15) | -0.054 (-0.18 to +0.07) | -0.043 (-0.18 to +0.10) |
| contrast (std) at 224 | -0.024 (-0.19 to +0.16) | -0.031 (-0.18 to +0.20) | +0.046 (-0.12 to +0.19) | +0.014 (-0.10 to +0.16) |
| dark fraction at 224 | +0.039 (-0.10 to +0.18) | +0.041 (-0.10 to +0.23) | -0.022 (-0.13 to +0.11) | -0.001 (-0.11 to +0.07) |
| grain at 224 | -0.167 (-0.27 to -0.05) | -0.143 (-0.30 to -0.01) | -0.117 (-0.25 to -0.01) | -0.083 (-0.34 to +0.04) |
| grain at 224, horizontal | -0.059 (-0.13 to +0.06) | -0.043 (-0.18 to +0.13) | - | - |
| grain at 224, vertical | -0.267 (-0.37 to -0.16) | -0.245 (-0.36 to -0.07) | - | - |

**Validation, PNEUMONIA images** - named: none

| Covariate | Spearman, seed 42 | Spearman, seed 43 | Partial, seed 42 | Partial, seed 43 |
|---|---|---|---|---|
| JPEG quantisation-table mean | +0.034 (-0.10 to +0.08) | +0.039 (-0.08 to +0.09) | -0.058 (-0.16 to -0.01) | -0.056 (-0.16 to +0.03) |
| file bytes per pixel | -0.071 (-0.21 to +0.11) | -0.098 (-0.30 to +0.04) | -0.044 (-0.12 to +0.09) | -0.054 (-0.20 to +0.08) |
| original megapixels | -0.357 (-0.55 to -0.21) | -0.342 (-0.56 to -0.19) | -0.131 (-0.37 to -0.04) | -0.162 (-0.37 to +0.01) |
| original aspect ratio (w/h) | +0.310 (+0.13 to +0.44) | +0.303 (+0.15 to +0.48) | +0.157 (+0.05 to +0.28) | +0.121 (+0.02 to +0.34) |
| mean intensity at 224 | -0.095 (-0.25 to +0.05) | -0.110 (-0.27 to +0.09) | -0.128 (-0.27 to +0.02) | -0.125 (-0.29 to +0.01) |
| contrast (std) at 224 | -0.158 (-0.39 to +0.06) | -0.186 (-0.33 to +0.04) | +0.088 (-0.01 to +0.17) | +0.071 (-0.03 to +0.24) |
| dark fraction at 224 | -0.107 (-0.33 to +0.05) | -0.162 (-0.31 to +0.09) | -0.087 (-0.20 to +0.01) | -0.081 (-0.15 to +0.02) |
| grain at 224 | -0.251 (-0.45 to -0.07) | -0.271 (-0.44 to -0.00) | -0.114 (-0.30 to +0.05) | -0.132 (-0.27 to +0.02) |
| grain at 224, horizontal | -0.129 (-0.32 to +0.00) | -0.140 (-0.32 to +0.07) | - | - |
| grain at 224, vertical | -0.348 (-0.55 to -0.13) | -0.376 (-0.53 to -0.09) | - | - |

## The #33 link: the healthy test images' margin increase from 20% to 100% (descriptive)

Per architecture, regime and seed, the Spearman correlation of each covariate with how much a healthy
test image's margin rose from the 20% run to the 100% run: median over the 9 pairs of a seed.

| Covariate | Seed 42 | Seed 43 |
|---|---|---|
| JPEG quantisation-table mean | - | - |
| file bytes per pixel | -0.013 (-0.22 to +0.20) | -0.002 (-0.17 to +0.15) |
| original megapixels | -0.337 (-0.68 to -0.11) | -0.325 (-0.64 to +0.17) |
| original aspect ratio (w/h) | +0.310 (+0.11 to +0.65) | +0.297 (-0.05 to +0.59) |
| mean intensity at 224 | -0.085 (-0.27 to +0.10) | -0.189 (-0.35 to -0.08) |
| contrast (std) at 224 | +0.016 (-0.11 to +0.07) | -0.024 (-0.20 to +0.10) |
| dark fraction at 224 | +0.099 (-0.13 to +0.25) | +0.142 (-0.14 to +0.37) |
| grain at 224 | -0.070 (-0.18 to +0.05) | +0.057 (-0.13 to +0.16) |

## Where the true-class maps sit (468 subset images per run)

Background cells are the 7x7 cells more than half of whose pixels are dark; edge cells are the top
corners and the bottom row. Median over the 18 runs of an architecture (both seeds), empty maps left
out and counted.

| Architecture | Class | Share on background cells | Share on top corners and bottom row | Peak on a background cell | Empty maps |
|---|---|---|---|---|---|
| MobileNetV3-Small | NORMAL | 0.022 | 0.079 | 0.002 | 4 |
| MobileNetV3-Small | PNEUMONIA | 0.068 | 0.228 | 0.045 | 0 |
| ResNet18 | NORMAL | 0.024 | 0.131 | 0.005 | 191 |
| ResNet18 | PNEUMONIA | 0.094 | 0.130 | 0.013 | 1 |
| EfficientNet-B0 | NORMAL | 0.045 | 0.074 | 0.011 | 9 |
| EfficientNet-B0 | PNEUMONIA | 0.068 | 0.354 | 0.045 | 0 |

**Paired comparisons** - second run minus first, per class; regime rows over 9 cells (architecture
and data size), data-size rows over 9 cells (architecture and regime):

| Comparison | Measure | Seed 42 up / equal / down | Seed 43 up / equal / down | Consistent + / - | Strict + / - | Median mean |
|---|---|---|---|---|---|---|
| mild-none, NORMAL | share of the map on background cells | 7 / 0 / 2 | 6 / 0 / 3 | 5 / 1 | 5 / 0 | +0.0060 |
| mild-none, NORMAL | share on the top corners and bottom row | 8 / 0 / 1 | 9 / 0 / 0 | 8 / 0 | 6 / 0 | +0.0362 |
| mild-none, NORMAL | peak on a background cell | 4 / 3 / 2 | 2 / 5 / 2 | 1 / 1 | 0 / 0 | +0.0021 |
| mild-none, PNEUMONIA | share of the map on background cells | 2 / 0 / 7 | 4 / 0 / 5 | 2 / 5 | 2 / 3 | -0.0061 |
| mild-none, PNEUMONIA | share on the top corners and bottom row | 0 / 0 / 9 | 0 / 0 / 9 | 0 / 9 | 0 / 9 | -0.0734 |
| mild-none, PNEUMONIA | peak on a background cell | 3 / 2 / 4 | 3 / 3 / 3 | 1 / 0 | 0 / 0 | -0.0085 |
| strong-none, NORMAL | share of the map on background cells | 7 / 0 / 2 | 7 / 0 / 2 | 6 / 1 | 6 / 1 | +0.0148 |
| strong-none, NORMAL | share on the top corners and bottom row | 8 / 0 / 1 | 8 / 0 / 1 | 7 / 0 | 5 / 0 | +0.0686 |
| strong-none, NORMAL | peak on a background cell | 3 / 3 / 3 | 4 / 4 / 1 | 2 / 1 | 0 / 0 | +0.0021 |
| strong-none, PNEUMONIA | share of the map on background cells | 2 / 0 / 7 | 0 / 0 / 9 | 0 / 7 | 0 / 6 | -0.0231 |
| strong-none, PNEUMONIA | share on the top corners and bottom row | 0 / 0 / 9 | 0 / 0 / 9 | 0 / 9 | 0 / 9 | -0.0929 |
| strong-none, PNEUMONIA | peak on a background cell | 1 / 0 / 8 | 1 / 0 / 8 | 0 / 7 | 0 / 3 | -0.0299 |
| strong-mild, NORMAL | share of the map on background cells | 6 / 0 / 3 | 8 / 0 / 1 | 6 / 1 | 3 / 1 | +0.0132 |
| strong-mild, NORMAL | share on the top corners and bottom row | 6 / 0 / 3 | 8 / 0 / 1 | 5 / 0 | 4 / 0 | +0.0138 |
| strong-mild, NORMAL | peak on a background cell | 4 / 0 / 5 | 3 / 5 / 1 | 2 / 0 | 0 / 0 | -0.0021 |
| strong-mild, PNEUMONIA | share of the map on background cells | 3 / 0 / 6 | 1 / 0 / 8 | 0 / 5 | 0 / 5 | -0.0223 |
| strong-mild, PNEUMONIA | share on the top corners and bottom row | 3 / 0 / 6 | 0 / 0 / 9 | 0 / 6 | 0 / 6 | -0.0492 |
| strong-mild, PNEUMONIA | peak on a background cell | 1 / 1 / 7 | 2 / 0 / 7 | 1 / 7 | 0 / 3 | -0.0363 |
| 50%-20%, NORMAL | share of the map on background cells | 4 / 0 / 5 | 7 / 0 / 2 | 3 / 1 | 1 / 0 | -0.0014 |
| 50%-20%, NORMAL | share on the top corners and bottom row | 7 / 0 / 2 | 5 / 0 / 4 | 3 / 0 | 2 / 0 | +0.0053 |
| 50%-20%, NORMAL | peak on a background cell | 4 / 3 / 2 | 2 / 5 / 2 | 2 / 0 | 0 / 0 | +0.0000 |
| 50%-20%, PNEUMONIA | share of the map on background cells | 5 / 0 / 4 | 4 / 0 / 5 | 2 / 2 | 2 / 1 | -0.0017 |
| 50%-20%, PNEUMONIA | share on the top corners and bottom row | 4 / 0 / 5 | 4 / 0 / 5 | 0 / 1 | 0 / 0 | +0.0017 |
| 50%-20%, PNEUMONIA | peak on a background cell | 5 / 0 / 4 | 4 / 0 / 5 | 1 / 1 | 0 / 1 | -0.0043 |
| 100%-20%, NORMAL | share of the map on background cells | 9 / 0 / 0 | 7 / 0 / 2 | 7 / 0 | 4 / 0 | +0.0114 |
| 100%-20%, NORMAL | share on the top corners and bottom row | 7 / 0 / 2 | 6 / 0 / 3 | 5 / 1 | 5 / 1 | +0.0443 |
| 100%-20%, NORMAL | peak on a background cell | 7 / 1 / 1 | 4 / 3 / 2 | 3 / 0 | 0 / 0 | +0.0023 |
| 100%-20%, PNEUMONIA | share of the map on background cells | 4 / 0 / 5 | 5 / 0 / 4 | 2 / 2 | 1 / 2 | -0.0044 |
| 100%-20%, PNEUMONIA | share on the top corners and bottom row | 4 / 0 / 5 | 5 / 0 / 4 | 2 / 2 | 2 / 2 | -0.0087 |
| 100%-20%, PNEUMONIA | peak on a background cell | 5 / 0 / 4 | 4 / 1 / 4 | 2 / 2 | 0 / 1 | +0.0043 |
