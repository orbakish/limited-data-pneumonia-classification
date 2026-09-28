# What the project shows so far

*Current as of 2026-09-28, branch `yotam/analysis-followup`.* The bottom lines of the work to date, for Or and
Yotam, and a starting point for the report. Nothing here is new: every number comes from
`results/analysis/summary.md`, `results/analysis/runs.csv` or a row of `NOTES.md`, named beside it so it can be
checked. Readings that still wait for Or's agreement are marked **(to confirm)**.

**How to read the numbers.** Every result compares two trained models on the same 624 test images. A *cell* is
one architecture at one data size or in one augmentation regime. *Consistent* means both seeds point the same
way; *strict* means both seeds' 95% intervals also exclude 0. A finding counts as strong only when it is
consistent (NOTES.md #32). The intervals cover test-sampling noise; retraining with another seed is a second
noise, often as large as the effects (section 5).

## The short version

1. Augmentation - strong augmentation above all - improves the models in every cell of both seeds, at every data
   size, and the clearest gain is fewer false alarms. Its ranking gain does not grow when data is scarcer
   **(to confirm, #35)**.
2. It does not harm explanation stability. It makes explanations more stable under the rotation it trained on,
   and leaves stability under noise where it was.
3. More training data did not help on the official test set: the models called more healthy images pneumonia.
   The cut-off set on our validation set does not carry over to the test set **(to confirm, #33)**.
4. The three architectures rank test images about equally well and respond alike to more data; they differ in
   how much augmentation helps them **(to confirm, #34)**.
5. The models take fine noise as evidence of pneumonia, and their Grad-CAM maps sometimes point at image edges
   and markers. The shortcut check (#15) is still to come.

## 1. Does augmentation improve performance under limited data?

**Yes - strong augmentation in every cell of both seeds, mild less clearly. The clearest gain is fewer false
alarms.**

- Strong vs none: ROC-AUC higher in 9 of 9 cells in both seeds, strict in 7, median +0.020. Mild vs none: 7 and
  8 of 9, strict in 2, +0.010. Strong vs mild: +0.009, strict in 2 (NOTES.md, Reference numbers, results of the
  two sweeps).
- Specificity at 0.5 is where it shows: strong vs none +0.088, 9 of 9 in both seeds, strict in 5. Recall hardly
  moves (+0.001), because it is already 0.97-0.997 in every run.
- EfficientNet-B0 gains the most (+0.029 mild, +0.032 strong), ResNet18 the least (#34).
- **When - more at 20% than at 100%? (to confirm, #35)** Not more when data is scarce. Strong augmentation raises
  ROC-AUC at every size - in 3 of 3 architectures in both seeds at 20%, 50% and 100% - and the gain does not
  consistently grow or shrink with more data: only 1 of the 12 paired comparisons (effect at the larger size minus
  at 20%) is strict, EfficientNet-B0 strong at 100% (+0.023), where the strongly augmented run itself kept
  improving with more data. What does change with size is the operating point: augmentation's gain in specificity
  is largest at 100% (strong: +0.079, +0.143, +0.188), because more data pushes the unaugmented models' healthy test
  images towards the cut-off and augmentation largely prevents it (section 4). `summary.md`, section "Augmentation
  by data size", lists all 12 comparisons with both seeds' intervals.

## 2. Does stronger augmentation harm explanation stability?

**No. Augmentation raises the stability gap, through stability under the rotation it trained on - not under
noise.**

- The gap (in-family minus out-of-family Pearson) grows with strong augmentation in 5 of 9 cells in both seeds
  and falls in none, strict in all 5, median +0.064. Mild: 5 up, 1 down, +0.051.
- It grows through the in-family score (+0.066), while the out-of-family score does not move (-0.004).
  Augmentation buys invariance to the perturbation family it trained on, not stability in general - the reason
  the project measures two families (NOTES.md, the stability measurement row).
- It holds on the correctly classified images only (7 up, 0 down), with an empty map counted as 0, and after
  the chance adjustment, which never changes the gap's sign. So neither the misclassified images, nor ResNet18's
  empty maps, nor the chance baseline produce it.
- By architecture: MobileNetV3-Small shows it cleanly (strict in every cell, +0.087 mild, +0.089 strong),
  ResNet18 hardly at all (-0.020 / +0.001), and for EfficientNet-B0 it depends on the seed (#34).
- **Caveat:** the perturbation sizes are provisional, to be reviewed together (#29).

## 3. Do lightweight and larger models behave differently? (to confirm, #34)

**Not in how well they rank test images, and not in their response to more data - but in their response to
augmentation.**

- No architecture ranks test images consistently better at any data size. The ROC-AUC medians lie between
  -0.009 and +0.010, and no comparison has more than 2 of 3 cells agreeing in both seeds - the size of
  retraining with another seed.
- ResNet18 differs in where its decisions fall: more recall and less specificity at 0.5 (at 50% data, -0.10 and
  -0.11 against the other two).
- More data: none of the three gains test ROC-AUC.
- Augmentation: EfficientNet-B0 gains the most ranking and ResNet18 the least; the lightweight
  MobileNetV3-Small is the only model whose stability gap grows with augmentation in every cell of both seeds.

## 4. More training data did not help on the official test set (to confirm, #33)

**From 20% to 100% of the training data, test ROC-AUC has no consistent direction and specificity falls - at
0.5 and at the tuned threshold alike - while every validation measure improves.**

- Test specificity at the tuned threshold falls in 17 of 18 comparisons (seed 42) and 13 of 18 (seed 43).
  Validation specificity rises: up / equal / down 16 / 2 / 0 and 15 / 3 / 0.
- Without any threshold: with more data, the healthy test images move towards being called pneumonia (median
  margin change +0.94 and +0.86; consistent 10 up, 0 down), and the healthy validation images do not.
- In every one of the 54 runs, the most pneumonia-looking tenth of the healthy images is safely NORMAL on
  validation and called pneumonia on test (90th percentile of their margins: -7.5 to -1.2 against +3.1 to
  +16.8; `runs.csv`).
- The ranking survives: a cut-off chosen on the test set would give balanced accuracy 0.871-0.942 instead of
  0.715-0.877 at 0.5. **This is a diagnostic only, never a result**, because that cut-off is chosen on the test
  set.
- Longer training does not explain it (the epochs trained have no direction), and neither does class balance
  (specificity counts NORMAL images only).
- **Augmentation largely prevents it (#35).** From 20% to 100%, the unaugmented MobileNetV3-Small's and
  EfficientNet-B0's healthy test images move about +2.2 towards the cut-off, the strongly augmented
  EfficientNet-B0's -0.09; the specificity comparisons that are strict are all at 100%, where the unaugmented runs
  change by -0.07 to -0.16 and the augmented ones by -0.05 to +0.02.
- **Reading (to confirm):** an operating-point shift between the training distribution, which our validation
  set samples, and the official test set, growing with more data. It ties to #15. Figure:
  `results/analysis/analysis_operating_point_shift.png`.

## 5. What we learned about the method

- **The validation set is at its ceiling.** ROC-AUC 0.994-1.000 and specificity 0.955-1.000 in every run, so it
  can barely tell the runs apart (`runs.csv`). During tuning this moved checkpoint selection from ROC-AUC to
  balanced accuracy (NOTES.md, training recipe). In the sweep, five checkpoints came from epoch 1 or 2, and a
  threshold chosen on validation gave no consistent advantage on test: better / worse / equal in 5 / 15 / 7
  experiments (seed 42) and 10 / 11 / 6 (seed 43) (#31).
- **Retraining is as noisy as many of the effects.** Seed 43 minus seed 42 moves test ROC-AUC by a median 0.006
  (90th percentile 0.023) and the stability gap by 0.047 - outside a single run's interval in 11 and 23 of the 27
  experiments. A single seed's intervals would have overstated several findings; hence the two-seed rule (#32).
- **The models read fine noise as pneumonia.** Noise too faint to see flips 5,272 test predictions from NORMAL
  to PNEUMONIA and 28 the other way (seed 42), in every architecture and regime, and it flipped validation
  predictions at similar rates during the calibration (#7). The errors lean the same way: 2,267 false alarms
  against 90 misses at 0.5. Together with the Grad-CAM examples - the evidence for pneumonia often at image
  edges, markers and wires - this is what the shortcut check (#15) is for.
- **Grad-CAM similarity has a floor well above 0.** The maps of two different same-class images correlate at
  0.34-0.68 (the chance baseline, seed 42), so stability is read against that baseline, and the gap is the robust
  summary (NOTES.md, how the stability results are read).
- **ResNet18's Grad-CAM maps often go empty under noise** - up to 93 of the 468 images in one run - which is why
  the stability comparisons also score an empty map as 0.
- **The pipeline reproduces itself exactly.** All 54 checkpoints re-score to their stored results with a
  difference of 0.0, and the analysis reproduces the stored intervals. Every run stopped by early stopping, the
  longest after 27 of 30 epochs.
- **Small implementation details can change results.** MobileNetV3-Small's BatchNorm momentum made full
  fine-tuning look worse than it was (#26, the T16 test); partial fine-tuning avoided it and became the recipe
  (#6).

## 6. Limitations the report should state

- One dataset. The test set comes from the same sources, and the two classes differ in image size, aspect
  ratio, framing and attached devices (#15); squashing to 224x224 cannot remove the aspect-ratio difference.
- Patient grouping comes from filenames: the NORMAL groups are case keys, not proven patients, and `person<N>`
  merges children who are probably distinct (NOTES.md, the patient-key row).
- Two seeds per configuration; the intervals cover test sampling, not training.
- The validation set is at its ceiling and does not show the test set's shift (sections 4 and 5).
- The stability perturbations are provisional (#29), and the metric changed from the proposal's cosine
  similarity to Pearson correlation - the report has to say so and why (#30).
- Class weighting and the validation-chosen threshold correct the training imbalance, not the test set's
  operating point (#17, #27).

## 7. Still open

- #33, #34 and #35 - Or to confirm the readings above.
- #29 - the perturbation sizes, reviewed together.
- #15 - the shortcut check, planned next.
- `experiments/experiment_log.md` - the results rows (Or).
- The report states the change from cosine to Pearson (#30).
