# Shortcut check (NOTES.md #15) - covariates and regions

Written by `python -m src.shortcut` from the dataset and `results/runs/`; change the code, not this file.
The readings were stated in the plan before any model output was related to an image property (NOTES.md
#15). The intervention (Colab) follows the regions, from the section "The intervention (part 3)" on; the
visual audit's labels and their check close it, in "The visual audit (part 4)".

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

## The intervention (part 3)

Every checkpoint scored validation and test again on the Colab T4 that collected the scores
(`python -m src.shortcut --intervene`, `results/runs/*/shortcut.json`) in four versions: unperturbed,
re-encoded as a JPEG of quality 75 (the healthy test images' encoding), blurred (Gaussian, sigma 1 px) and
sharpened (the image plus what the blur removes: the blur's pixel change before the clip to 0-1, less
after it). *delta* is the median over a class's images of each image's margin change against the
unperturbed version, in logits; the margin is the PNEUMONIA logit minus the NORMAL logit, and above 0 is
called pneumonia at 0.5.

**Checks:**

- The images, labels and unperturbed logits of all 54 runs equal the collected ones exactly.
- The margin maps are read in the stability subset's order, which equals every run's stability records.
- Test specificity at 0.5 from the unperturbed logits against every results file: largest difference 1.1e-16.

**Encodings in the data** (`covariates.csv`; a luminance-table mean of 29.03 is quality 75, a lower mean a
higher quality):

| Split, class | Images | Quality 75 | Higher quality | Lower quality |
|---|---|---|---|---|
| validation, NORMAL | 202 | 0 | 202 | 0 |
| validation, PNEUMONIA | 583 | 568 | 15 | 0 |
| test, NORMAL | 234 | 234 | 0 | 0 |
| test, PNEUMONIA | 390 | 352 | 38 | 0 |

### The readings, stated before any perturbed score was read (NOTES.md #15)

- **Encoding:** supported if, on validation, delta(NORMAL) >= 1 logit and >= 5 x |delta(PNEUMONIA)| in at least 12 of the 18 runs of every architecture; not supported if delta(NORMAL) < 0.5 in at least 12 of the 18 runs of every architecture; otherwise stated per architecture.
- **Texture or anything unusual:** sharpening moves the healthy validation images towards NORMAL (delta < 0) in at least 12 of an architecture's 18 runs - texture; towards PNEUMONIA (delta > 0) in at least 12 - anything unusual looks like pneumonia; otherwise neither. A delta of exactly 0 counts for neither. Read on validation, test beside it (decided on 2026-09-28, before any number was seen); a different word on test is a disagreement.
- An architecture's 18 runs are 9 configurations x 2 seeds; the two seeds of a configuration are not independent, so a count is not 18 independent votes.

**Encoding: not supported**.

| Architecture | Runs | Encoding: >= 1 and >= 5x | Encoding: < 0.5 | Encoding | Sharpen, validation: < 0 / > 0 / = 0 | Validation | Sharpen, test: < 0 / > 0 / = 0 | Test | Blur, validation: > 0 |
|---|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Small | 18 | 0 | 18 | not supported | 2 / 16 / 0 | anything unusual | 4 / 14 / 0 | anything unusual | 18 |
| ResNet18 | 18 | 0 | 18 | not supported | 5 / 13 / 0 | anything unusual | 6 / 12 / 0 | anything unusual | 17 |
| EfficientNet-B0 | 18 | 0 | 18 | not supported | 1 / 17 / 0 | anything unusual | 2 / 16 / 0 | anything unusual | 17 |

Sharpen, test against validation: the same verdict for every architecture. Pixel change of the healthy validation images: blur 0.0100, sharpen 0.0097 (equal before the clip).

**delta per architecture** - median over its runs (lowest to highest):

| Version | Split, class | MobileNetV3-Small | ResNet18 | EfficientNet-B0 |
|---|---|---|---|---|
| reencoded | validation, NORMAL | +0.082 (+0.04 to +0.20) | +0.011 (-0.01 to +0.04) | +0.061 (+0.02 to +0.11) |
| reencoded | validation, PNEUMONIA | +0.000 (-0.00 to +0.01) | +0.002 (+0.00 to +0.01) | +0.001 (-0.01 to +0.00) |
| reencoded | test, NORMAL | +0.000 (-0.00 to +0.00) | +0.000 (-0.00 to +0.00) | +0.001 (-0.00 to +0.00) |
| reencoded | test, PNEUMONIA | +0.000 (-0.00 to +0.00) | +0.000 (-0.00 to +0.00) | -0.001 (-0.01 to +0.01) |
| blur | validation, NORMAL | +6.253 (+2.89 to +13.80) | +1.615 (-0.29 to +4.60) | +2.141 (-0.16 to +8.21) |
| blur | validation, PNEUMONIA | +0.659 (-1.02 to +4.68) | -2.546 (-4.98 to -0.15) | -4.155 (-9.69 to -1.53) |
| blur | test, NORMAL | +4.529 (+1.50 to +7.79) | +0.621 (-1.31 to +4.01) | +0.480 (-1.58 to +4.16) |
| blur | test, PNEUMONIA | +0.654 (-1.16 to +5.57) | -3.047 (-6.01 to -1.13) | -4.439 (-10.54 to -1.80) |
| sharpen | validation, NORMAL | +0.822 (-0.28 to +3.80) | +0.358 (-2.07 to +1.92) | +0.879 (-0.03 to +2.17) |
| sharpen | validation, PNEUMONIA | -0.932 (-3.39 to +0.36) | +1.396 (-1.64 to +4.68) | -1.439 (-3.40 to -0.14) |
| sharpen | test, NORMAL | +1.049 (-1.13 to +2.10) | +0.448 (-2.53 to +2.00) | +0.515 (-0.13 to +1.76) |
| sharpen | test, PNEUMONIA | -1.108 (-3.72 to +0.40) | +1.803 (-1.13 to +5.43) | -1.616 (-3.54 to -0.31) |

**What the re-encode alone cannot separate.** The encoding and "anything unusual looks like pneumonia" both predict that re-encoding raises the healthy margins; how far each version moves the pixels tells them apart. Healthy validation images: median delta over an architecture's runs / median pixel change (0-1 units) = delta per unit of pixel change:

| Architecture | reencoded | blur | sharpen |
|---|---|---|---|
| MobileNetV3-Small | +0.08 / 0.0008 = +109 | +6.25 / 0.0100 = +627 | +0.82 / 0.0097 = +84 |
| ResNet18 | +0.01 / 0.0008 = +14 | +1.62 / 0.0100 = +162 | +0.36 / 0.0097 = +37 |
| EfficientNet-B0 | +0.06 / 0.0008 = +81 | +2.14 / 0.0100 = +215 | +0.88 / 0.0097 = +90 |

- MobileNetV3-Small: the re-encode moves the healthy validation pixels less than the blur (0.0008 against 0.0100) and their margins less (+0.08 against +6.25) - on this comparison, not told apart from a response to any change.
- ResNet18: the re-encode moves the healthy validation pixels less than the blur (0.0008 against 0.0100) and their margins less (+0.01 against +1.62) - on this comparison, not told apart from a response to any change.
- EfficientNet-B0: the re-encode moves the healthy validation pixels less than the blur (0.0008 against 0.0100) and their margins less (+0.06 against +2.14) - on this comparison, not told apart from a response to any change.

**How much of the validation-to-test gap in specificity the re-encode reproduces** - descriptive, not a second verdict: a re-encode that moves the healthy margins by 1 logit or more need not carry them across the cut-off when they sit far below it. Specificity at 0.5 on validation (s_v), after the re-encode (s_r) and on test (s_t); false alarms among the healthy images; the share is (s_v - s_r) / (s_v - s_t), not clipped, and '-' when s_v - s_t < 0.05. Median over an architecture's runs (the share: lowest to highest):

| Architecture | s_v | s_r | s_t | False alarms: validation / re-encoded / test | Share | Runs with a share |
|---|---|---|---|---|---|---|
| MobileNetV3-Small | 0.985 | 0.985 | 0.688 | 3 / 3 / 73 | +0.000 (+0.00 to +0.04) | 18 of 18 |
| ResNet18 | 0.985 | 0.980 | 0.626 | 3 / 4 / 88 | +0.000 (-0.02 to +0.02) | 18 of 18 |
| EfficientNet-B0 | 0.990 | 0.990 | 0.677 | 2 / 2 / 76 | +0.000 (-0.02 to +0.05) | 18 of 18 |

**The test re-encode, a sanity check.** Every healthy test image is already quality 75, so re-encoding changes it little; re-saving a quality-75 file at 75 still quantises it again, so it is not a no-op - read beside its pixel change. Median delta over an architecture's runs:

| | Validation, NORMAL | Test, NORMAL | Test, PNEUMONIA | Test, PNEUMONIA at quality 75 |
|---|---|---|---|---|
| MobileNetV3-Small | +0.082 | +0.000 | +0.000 | +0.000 |
| ResNet18 | +0.011 | +0.000 | +0.000 | -0.000 |
| EfficientNet-B0 | +0.061 | +0.001 | -0.001 | -0.000 |
| Pixel change | 0.0008 | 0.0000 | 0.0000 | 0.0000 |

**The healthy validation images' re-encode delta by regime and data size** (descriptive) - seed 42 / seed 43:

| Architecture | Regime | 20% | 50% | 100% |
|---|---|---|---|---|
| MobileNetV3-Small | None | +0.06 / +0.09 | +0.13 / +0.15 | +0.20 / +0.12 |
| MobileNetV3-Small | Mild | +0.05 / +0.06 | +0.08 / +0.11 | +0.07 / +0.09 |
| MobileNetV3-Small | Strong | +0.06 / +0.09 | +0.04 / +0.06 | +0.09 / +0.06 |
| ResNet18 | None | +0.01 / +0.04 | -0.01 / +0.04 | +0.03 / +0.01 |
| ResNet18 | Mild | -0.01 / -0.00 | +0.03 / +0.01 | +0.01 / +0.02 |
| ResNet18 | Strong | +0.03 / -0.01 | +0.01 / -0.00 | +0.03 / +0.01 |
| EfficientNet-B0 | None | +0.06 / +0.08 | +0.11 / +0.06 | +0.08 / +0.11 |
| EfficientNet-B0 | Mild | +0.07 / +0.02 | +0.07 / +0.06 | +0.07 / +0.06 |
| EfficientNet-B0 | Strong | +0.04 / +0.03 | +0.06 / +0.03 | +0.04 / +0.04 |

**The #33 link** (descriptive): the Spearman correlation of the healthy validation images' re-encode delta with the run's healthy test images' median margin, over a seed's runs; and, closer to #33, of the change of that delta from 20% to 100% with the change of that margin, over the architecture x regime cells.

| Seed | Across the runs | Change from 20% to 100% |
|---|---|---|
| 42 | -0.17 (27 runs) | +0.48 (9 cells) |
| 43 | -0.05 (27 runs) | +0.42 (9 cells) |

## Where the margin maps sit (468 subset images per run)

A margin map explains the PNEUMONIA logit minus the NORMAL logit, so on a healthy image it shows the
evidence for pneumonia - where a false alarm's evidence sits. Measures as for the true-class maps; median
over the 18 runs of an architecture (both seeds), empty maps left out and counted.

| Architecture | Class | Share on background cells | Share on top corners and bottom row | Peak on a background cell | Empty maps |
|---|---|---|---|---|---|
| MobileNetV3-Small | NORMAL | 0.138 | 0.316 | 0.083 | 1 |
| MobileNetV3-Small | PNEUMONIA | 0.068 | 0.224 | 0.045 | 0 |
| ResNet18 | NORMAL | 0.239 | 0.293 | 0.233 | 178 |
| ResNet18 | PNEUMONIA | 0.096 | 0.136 | 0.017 | 0 |
| EfficientNet-B0 | NORMAL | 0.152 | 0.534 | 0.144 | 241 |
| EfficientNet-B0 | PNEUMONIA | 0.068 | 0.352 | 0.047 | 0 |

**Paired comparisons** - second run minus first, per class; regime rows over 9 cells (architecture
and data size), data-size rows over 9 cells (architecture and regime):

| Comparison | Measure | Seed 42 up / equal / down | Seed 43 up / equal / down | Consistent + / - | Strict + / - | Median mean |
|---|---|---|---|---|---|---|
| mild-none, NORMAL | share of the map on background cells | 4 / 0 / 5 | 6 / 0 / 3 | 3 / 2 | 2 / 0 | +0.0024 |
| mild-none, NORMAL | share on the top corners and bottom row | 0 / 0 / 9 | 0 / 0 / 9 | 0 / 9 | 0 / 7 | -0.0958 |
| mild-none, NORMAL | peak on a background cell | 4 / 1 / 4 | 5 / 0 / 4 | 3 / 3 | 1 / 0 | +0.0004 |
| mild-none, PNEUMONIA | share of the map on background cells | 2 / 0 / 7 | 4 / 0 / 5 | 2 / 5 | 2 / 3 | -0.0069 |
| mild-none, PNEUMONIA | share on the top corners and bottom row | 0 / 0 / 9 | 0 / 0 / 9 | 0 / 9 | 0 / 9 | -0.0713 |
| mild-none, PNEUMONIA | peak on a background cell | 3 / 1 / 5 | 5 / 2 / 2 | 2 / 1 | 0 / 0 | +0.0000 |
| strong-none, NORMAL | share of the map on background cells | 2 / 0 / 7 | 3 / 0 / 6 | 1 / 5 | 0 / 4 | -0.0198 |
| strong-none, NORMAL | share on the top corners and bottom row | 1 / 0 / 8 | 1 / 0 / 8 | 0 / 7 | 0 / 7 | -0.1535 |
| strong-none, NORMAL | peak on a background cell | 5 / 0 / 4 | 3 / 0 / 6 | 2 / 3 | 0 / 2 | -0.0170 |
| strong-none, PNEUMONIA | share of the map on background cells | 2 / 0 / 7 | 0 / 0 / 9 | 0 / 7 | 0 / 6 | -0.0228 |
| strong-none, PNEUMONIA | share on the top corners and bottom row | 0 / 0 / 9 | 0 / 0 / 9 | 0 / 9 | 0 / 9 | -0.0896 |
| strong-none, PNEUMONIA | peak on a background cell | 1 / 1 / 7 | 1 / 0 / 8 | 0 / 6 | 0 / 3 | -0.0278 |
| strong-mild, NORMAL | share of the map on background cells | 2 / 0 / 7 | 2 / 0 / 7 | 1 / 6 | 0 / 4 | -0.0367 |
| strong-mild, NORMAL | share on the top corners and bottom row | 2 / 0 / 7 | 2 / 0 / 7 | 0 / 5 | 0 / 4 | -0.0242 |
| strong-mild, NORMAL | peak on a background cell | 3 / 0 / 6 | 2 / 1 / 6 | 1 / 5 | 0 / 2 | -0.0598 |
| strong-mild, PNEUMONIA | share of the map on background cells | 3 / 0 / 6 | 1 / 0 / 8 | 0 / 5 | 0 / 5 | -0.0217 |
| strong-mild, PNEUMONIA | share on the top corners and bottom row | 2 / 0 / 7 | 0 / 0 / 9 | 0 / 7 | 0 / 6 | -0.0468 |
| strong-mild, PNEUMONIA | peak on a background cell | 0 / 2 / 7 | 2 / 0 / 7 | 0 / 7 | 0 / 2 | -0.0406 |
| 50%-20%, NORMAL | share of the map on background cells | 4 / 0 / 5 | 4 / 0 / 5 | 3 / 4 | 2 / 1 | +0.0054 |
| 50%-20%, NORMAL | share on the top corners and bottom row | 4 / 0 / 5 | 5 / 0 / 4 | 3 / 3 | 1 / 3 | +0.0088 |
| 50%-20%, NORMAL | peak on a background cell | 5 / 0 / 4 | 3 / 1 / 5 | 3 / 4 | 1 / 2 | +0.0098 |
| 50%-20%, PNEUMONIA | share of the map on background cells | 5 / 0 / 4 | 5 / 0 / 4 | 2 / 1 | 2 / 1 | -0.0027 |
| 50%-20%, PNEUMONIA | share on the top corners and bottom row | 5 / 0 / 4 | 4 / 0 / 5 | 0 / 0 | 0 / 0 | +0.0014 |
| 50%-20%, PNEUMONIA | peak on a background cell | 5 / 0 / 4 | 3 / 1 / 5 | 1 / 1 | 0 / 1 | +0.0000 |
| 100%-20%, NORMAL | share of the map on background cells | 5 / 0 / 4 | 5 / 0 / 4 | 4 / 3 | 2 / 1 | +0.0024 |
| 100%-20%, NORMAL | share on the top corners and bottom row | 4 / 0 / 5 | 4 / 0 / 5 | 2 / 3 | 2 / 3 | +0.0162 |
| 100%-20%, NORMAL | peak on a background cell | 6 / 0 / 3 | 4 / 0 / 5 | 4 / 3 | 3 / 1 | +0.0319 |
| 100%-20%, PNEUMONIA | share of the map on background cells | 4 / 0 / 5 | 5 / 0 / 4 | 2 / 2 | 1 / 2 | -0.0039 |
| 100%-20%, PNEUMONIA | share on the top corners and bottom row | 4 / 0 / 5 | 5 / 0 / 4 | 2 / 2 | 2 / 2 | -0.0113 |
| 100%-20%, PNEUMONIA | peak on a background cell | 5 / 0 / 4 | 3 / 2 / 4 | 1 / 2 | 0 / 2 | +0.0043 |

**The audit's margin overlays:** 3 of the 30 margin maps of its healthy images are empty; at most 7 are allowed (fewer than a quarter), so overlays 61-90 are drawn.

## Pixel change per version

The mean absolute change of each version against the unperturbed image at 224x224, in 0-1 units
(model-free, `pixel_changes.csv`); median over the images. Blur / sharpen is 1 before the clip to 0-1.

| Split, class | Re-encoded | Blur | Sharpen | Blur / sharpen |
|---|---|---|---|---|
| validation, NORMAL | 0.0008 | 0.0100 | 0.0097 | 1.02 |
| validation, PNEUMONIA | 0.0000 | 0.0073 | 0.0071 | 1.02 |
| test, NORMAL | 0.0000 | 0.0093 | 0.0091 | 1.02 |
| test, PNEUMONIA | 0.0000 | 0.0075 | 0.0074 | 1.02 |

## The visual audit (part 4)

Six runs - each architecture without and with strong augmentation, 100% data, seed 42 - and the same
10 subset images in each (5 NORMAL, 5 PNEUMONIA): 60 overlays of the true-class map, then 30 of the
healthy images' margin maps (61-90), shuffled and numbered (`audit/`). One rater (Yotam) named each
overlay's main hot spot without the key; he had seen the region statistics and each architecture's map
style, so the audit is not blind, and some rules were settled while labelling (NOTES.md, the audit's
labelling row). With one rater there is no kappa. Every overlay's run, labels and region values:
`audit/overlays.csv`.

**Checks:**

- `key.csv` equals the code's draw of the audit, overlay for overlay.
- The overlays noted "empty map" are exactly those whose map is empty: 63, 76, 88 - left out of every count.
- Every label is one of the seven places; the neck overlays (9, 10, 32, 35, 50, 51, 54, 56) are labelled diffuse-other.

**Groups of the main label:** a *shortcut place* is image edge-background, marker-text, device-wires - the frame, printed markers and attached devices #15 is about. Below the diaphragm is anatomy, not an artefact, so it is a group of its own. *Neck* is the overlays whose main spot is the neck; it overlaps the lungs' tops at 7x7, so the lung field is counted twice - with the neck off the lungs, and as their tops. *Other* is diffuse-other outside the neck.

**True-class maps, NORMAL images** - non-empty overlays per group of the main label (5 images per run):

| Architecture | Regime | Maps | lung field | mediastinum-heart | below the diaphragm | neck | shortcut place | other | On the lung field: neck off / as lung tops | On a shortcut place: main / main or second |
|---|---|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Small | None | 5 | 2 | 0 | 0 | 2 | 1 | 0 | 2 / 4 of 5 | 1 / 1 of 5 |
| MobileNetV3-Small | Strong | 5 | 0 | 0 | 0 | 3 | 1 | 1 | 0 / 3 of 5 | 1 / 1 of 5 |
| ResNet18 | None | 5 | 2 | 0 | 0 | 1 | 2 | 0 | 2 / 3 of 5 | 2 / 2 of 5 |
| ResNet18 | Strong | 5 | 2 | 0 | 1 | 1 | 1 | 0 | 2 / 3 of 5 | 1 / 2 of 5 |
| EfficientNet-B0 | None | 5 | 4 | 1 | 0 | 0 | 0 | 0 | 4 / 4 of 5 | 0 / 0 of 5 |
| EfficientNet-B0 | Strong | 5 | 3 | 1 | 0 | 1 | 0 | 0 | 3 / 4 of 5 | 0 / 0 of 5 |
| All | None | 15 | 8 | 1 | 0 | 3 | 3 | 0 | 8 / 11 of 15 | 3 / 3 of 15 |
| All | Strong | 15 | 5 | 1 | 1 | 5 | 2 | 1 | 5 / 10 of 15 | 2 / 3 of 15 |

**True-class maps, PNEUMONIA images** - non-empty overlays per group of the main label (5 images per run):

| Architecture | Regime | Maps | lung field | mediastinum-heart | below the diaphragm | neck | shortcut place | other | On the lung field: neck off / as lung tops | On a shortcut place: main / main or second |
|---|---|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Small | None | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 0 / 0 of 5 | 5 / 5 of 5 |
| MobileNetV3-Small | Strong | 5 | 0 | 2 | 0 | 0 | 3 | 0 | 0 / 0 of 5 | 3 / 5 of 5 |
| ResNet18 | None | 5 | 3 | 0 | 0 | 0 | 2 | 0 | 3 / 3 of 5 | 2 / 2 of 5 |
| ResNet18 | Strong | 5 | 4 | 0 | 0 | 0 | 1 | 0 | 4 / 4 of 5 | 1 / 3 of 5 |
| EfficientNet-B0 | None | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 0 / 0 of 5 | 5 / 5 of 5 |
| EfficientNet-B0 | Strong | 5 | 2 | 0 | 0 | 0 | 3 | 0 | 2 / 2 of 5 | 3 / 3 of 5 |
| All | None | 15 | 3 | 0 | 0 | 0 | 12 | 0 | 3 / 3 of 15 | 12 / 12 of 15 |
| All | Strong | 15 | 6 | 2 | 0 | 0 | 7 | 0 | 6 / 6 of 15 | 7 / 11 of 15 |

**Margin maps, NORMAL images - the evidence for pneumonia in a healthy image** - non-empty overlays per group of the main label (5 images per run):

| Architecture | Regime | Maps | lung field | mediastinum-heart | below the diaphragm | neck | shortcut place | other | On the lung field: neck off / as lung tops | On a shortcut place: main / main or second |
|---|---|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Small | None | 5 | 1 | 0 | 3 | 0 | 1 | 0 | 1 / 1 of 5 | 1 / 1 of 5 |
| MobileNetV3-Small | Strong | 5 | 1 | 2 | 0 | 0 | 2 | 0 | 1 / 1 of 5 | 2 / 2 of 5 |
| ResNet18 | None | 5 | 3 | 0 | 1 | 0 | 1 | 0 | 3 / 3 of 5 | 1 / 1 of 5 |
| ResNet18 | Strong | 3 (2 empty) | 1 | 2 | 0 | 0 | 0 | 0 | 1 / 1 of 3 | 0 / 0 of 3 |
| EfficientNet-B0 | None | 5 | 0 | 0 | 1 | 0 | 4 | 0 | 0 / 0 of 5 | 4 / 4 of 5 |
| EfficientNet-B0 | Strong | 4 (1 empty) | 1 | 0 | 0 | 0 | 3 | 0 | 1 / 1 of 4 | 3 / 3 of 4 |
| All | None | 15 | 4 | 0 | 5 | 0 | 6 | 0 | 4 / 4 of 15 | 6 / 6 of 15 |
| All | Strong | 12 (3 empty) | 3 | 4 | 0 | 0 | 5 | 0 | 3 / 3 of 12 | 5 / 5 of 12 |

**Augmentation, paired by image** (descriptive): the same image in the none and the strong run of one
architecture, pooled over the three, both maps non-empty. The pairs are 5 images x 3 architectures, and
the healthy images return in the margin rows, so they are not independent pairs and no test is given.
Part 2's strong-none rows are a pointer, not a match: they average all 468 images over 9 cells, both
seeds and all data sizes (cells consistently up / down, from its tables above).

| Maps, class | Pairs | Shortcut place: none only / strong only / both / neither | Lung field, neck off | Lung field, neck as lung tops | Part 2, strong-none (a pointer, not a match) |
|---|---|---|---|---|---|
| true class, NORMAL | 15 | 1 / 0 / 2 / 12 | 5 / 2 / 3 / 5 | 3 / 2 / 8 / 2 | edge 7 up / 0 down of 9; background 6 up / 1 down of 9 |
| true class, PNEUMONIA | 15 | 6 / 1 / 6 / 2 | 1 / 4 / 2 / 8 | 1 / 4 / 2 / 8 | edge 0 up / 9 down of 9; background 0 up / 7 down of 9 |
| margin, NORMAL | 12 | 0 / 1 / 4 / 7 | 2 / 1 / 2 / 7 | 2 / 1 / 2 / 7 | edge 0 up / 7 down of 9; background 1 up / 5 down of 9 |

### The check of part 2's statistics (the reading was stated before the key was opened)

Part 2's shares of each non-empty overlay's own map, for the overlays of a group of main labels against
those on the lung field; the AUC is the chance that an overlay of the group has the higher share, a tie
counting half. The main line is every shortcut place, pooled over both map types: the labels and a share
**agree** if its AUC is at least 0.75, **agree weakly** from 0.60, and **do not agree** below; '-' if either group has fewer than 3 overlays. It is also given per map
type: overlays 61-90 are all margin maps of healthy images, which sit on the edge more whatever their
label, and 1-60 hold most of the lung-field labels - so if the pooled line agrees while a map type alone
does not agree, the reading is **disagreement**. The single labels are descriptive: by the labelling
rules, image edge-background also holds spots on the body around the arms, which are not dark cells, and
a marker on the dark border is marker-text, so a single label can differ from a share for that alone.

| Group | Maps | Share | Overlays: group / lung field | Median: group / lung field | AUC | Reading |
|---|---|---|---|---|---|---|
| shortcut place | both | background cells | 35 / 29 | 0.071 / 0.030 | 0.65 | agree weakly |
| shortcut place | true class | background cells | 24 / 22 | 0.050 / 0.019 | 0.64 | agree weakly |
| shortcut place | margin | background cells | 11 / 7 | 0.209 / 0.072 | 0.77 | agree |
| shortcut place | both | top corners and bottom row | 35 / 29 | 0.242 / 0.082 | 0.85 | agree |
| shortcut place | true class | top corners and bottom row | 24 / 22 | 0.193 / 0.071 | 0.89 | agree |
| shortcut place | margin | top corners and bottom row | 11 / 7 | 0.575 / 0.222 | 0.90 | agree |
| image edge-background | both | background cells | 13 / 29 | 0.153 / 0.030 | 0.65 | agree weakly |
| marker-text | both | background cells | 14 / 29 | 0.102 / 0.030 | 0.72 | agree weakly |
| marker-text | both | top corners and bottom row | 14 / 29 | 0.250 / 0.082 | 0.90 | agree |
| below the diaphragm | both | top corners and bottom row | 6 / 29 | 0.383 / 0.082 | 0.96 | agree |

**The share on background cells: agree weakly. The share on the top corners and bottom row: agree.**

**Where each map peaks, against its label** - the like-for-like check, since the label names the main
hot spot, the peak. The peak cell's category (a background cell first, then the top corners and bottom
row, else other) per main label, and whether it is where the label expects it (fixed before the key was
opened; a marker on a dark corner may read background):

| Main label | Peak: background / edge / other | Expected | Agree |
|---|---|---|---|
| lung field | 0 / 1 / 28 | other | 28 of 29 |
| mediastinum-heart | 1 / 0 / 7 | other | 7 of 8 |
| image edge-background | 4 / 2 / 7 | background | 4 of 13 |
| marker-text | 2 / 8 / 4 | background or edge | 10 of 14 |
| device-wires | 0 / 2 / 6 | - | - |
| below the diaphragm | 0 / 6 / 0 | edge | 6 of 6 |
| neck | 0 / 0 / 8 | - | - |
| diffuse-other | 0 / 0 / 1 | - | - |

Over the labels with an expected peak: 55 of 70 peaks agree.

**Caveats.** The label names the main hot spot - the peak, as the eye sees an overlay scaled to its own
maximum - while the shares are mass over all 49 cells: a diffuse map peaking on the lung with a quarter
of its mass on the border is rightly lung field and rightly has a high background share, so a middling
AUC is expected even when both are right; the peak table is the like-for-like check. The overlays are not
independent - the same 10 images in six runs of one seed at 100% data - and the rater had seen the
region statistics, so agreement is two views of one map agreeing, not an independent validation.
