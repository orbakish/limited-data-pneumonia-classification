# What the project shows so far

*Current as of 2026-09-30, branch `yotam/close-questions`.* The bottom lines of the work to date, for Or and
Yotam, and a starting point for the report. Nothing here is new: every number comes from
`results/analysis/summary.md`, `results/analysis/runs.csv`, `results/shortcut/summary.md` or a row of `NOTES.md`,
named beside it so it can be checked. Or confirmed the readings of #33-#36 on 2026-09-29; each question number
points to the Decisions row in `NOTES.md` that closes it.

**How to read the numbers.** Every result compares two trained models on the same 624 test images. A *cell* is
one architecture at one data size or in one augmentation regime. *Consistent* means both seeds point the same
way; *strict* means both seeds' 95% intervals also exclude 0. A finding counts as strong only when it is
consistent (NOTES.md #32). The intervals cover test-sampling noise; retraining with another seed is a second
noise, often as large as the effects (section 6).

## The short version

1. Augmentation - strong augmentation above all - improves the models in every cell of both seeds, at every data
   size, and the clearest gain is fewer false alarms. Its ranking gain does not grow when data is scarcer
   (#35).
2. It does not harm explanation stability. It makes explanations more stable under the rotation it trained on,
   and leaves stability under noise where it was.
3. More training data did not help on the official test set: the models called more healthy images pneumonia.
   The cut-off set on our validation set does not carry over to the test set (#33).
4. The three architectures rank test images about equally well and respond alike to more data; they differ in
   how much augmentation helps them (#34).
5. The models do not follow the JPEG encoding that separates the classes in training. But healthy images move
   towards pneumonia under blur, sharpening and added noise, the healthy test images' margins follow their
   original size, and the pneumonia evidence of two of the three architectures leans to the image's corners and
   bottom row - less so with augmentation (#36).

## 1. Does augmentation improve performance under limited data?

**Yes - strong augmentation in every cell of both seeds, mild less clearly. The clearest gain is fewer false
alarms.**

- Strong vs none: ROC-AUC higher in 9 of 9 cells in both seeds, strict in 7, median +0.020. Mild vs none: 7 and
  8 of 9, strict in 2, +0.010. Strong vs mild: +0.009, strict in 2 (NOTES.md, Reference numbers, results of the
  two sweeps).
- Specificity at 0.5 is where it shows: strong vs none +0.088, 9 of 9 in both seeds, strict in 5. Recall hardly
  moves (+0.001), because it is already 0.97-0.997 in every run.
- EfficientNet-B0 gains the most (+0.029 mild, +0.032 strong), ResNet18 the least (#34).
- **When - more at 20% than at 100%? (#35)** Not more when data is scarce. Strong augmentation raises
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
- The perturbation sizes - rotation +/-5 degrees, noise sigma 0.02 - are the ones fixed before the sweeps, kept
  after the results as the main measure (#29).

## 3. Do lightweight and larger models behave differently? (#34)

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

## 4. More training data did not help on the official test set (#33)

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
- **Reading:** an operating-point shift between the training distribution, which our validation
  set samples, and the official test set, growing with more data. It ties to #15; the JPEG encoding does not
  explain it (section 5). Figure:
  `results/analysis/analysis_operating_point_shift.png`.

## 5. Do the models take shortcuts? (#36)

**Not the JPEG encoding. But healthy images read as more like pneumonia after blur, sharpening or added noise,
the healthy test images' margins follow their size, and the evidence for pneumonia leans to the image's corners and
bottom row - less so with augmentation.**

- **Encoding.** In training the healthy images were saved at about JPEG quality 95 and the pneumonia images at
  75, so the quality table alone separates the classes (ROC-AUC 0.999); every healthy test image is at 75. Under
  the reading fixed in advance, the models do not follow it: re-encoding the healthy validation images to quality
  75 moves their margins by a median +0.01 to +0.08 logit per architecture (under 0.5 in 18 of 18 runs of each)
  and reproduces none of the validation-to-test drop in specificity (median share 0.000). The test copies the
  quality table, not everything else that may differ in how the test files were made.
- **Anything unusual, for healthy images.** Blur raises the healthy images' margins in 18 / 17 / 17 of 18 runs
  and sharpening in 16 / 13 / 17 (MobileNetV3-Small / ResNet18 / EfficientNet-B0; the sharpening verdict is the
  same on test), as added noise does. Pneumonia images mostly move the other way: blur and sharpening each push
  them towards NORMAL in two of the three architectures. So a healthy image that looks unusual reads as
  pneumonia; a pneumonia image does not read as more so.
- **Texture.** The healthy images are the grainier at 224x224 (grain as a lone pneumonia classifier: ROC-AUC
  0.087), and blur and sharpening both raise the healthy margins, so the models do not follow the dataset's
  texture.
- **Size.** Within the healthy test images, smaller originals look more like pneumonia (partial Spearman -0.30 /
  -0.32 in the two seeds, the only covariate named by the rule fixed in advance), and pneumonia originals are the
  smaller ones in the data. Within the healthy validation images size is not followed; aspect ratio is (+0.29 /
  +0.27). This is correlational - size was never changed in a test - and may reflect the test set's wider size
  spread as much as the models. With more data, the healthy test images that moved most towards pneumonia are the
  smaller and wider originals (descriptive; section 4).
- **Where the evidence sits.** The pneumonia maps put 0.228 and 0.354 of their mass on the top corners and
  bottom row (MobileNetV3-Small, EfficientNet-B0; an even spread would put 0.184 there, and ResNet18 puts 0.130),
  and augmentation lowers it in 9 of 9 cells in both seeds, strict in all 9. The healthy images' maps move
  slightly the other way (+0.0686 with strong augmentation, strict in 5 of 9). The evidence behind a false alarm -
  Grad-CAM of the pneumonia-minus-normal margin on healthy images - sits there far more (0.29-0.53), and
  augmentation lowers it too (strict in 7 of 9).
- **The visual audit** (6 runs of one seed, the same 10 images, one rater who had seen these statistics):
  without augmentation the main pneumonia hot spot is on the frame, a marker or a device in 12 of 15 maps, with
  strong augmentation in 7 (11 when a second label counts). In the false-alarm maps, the evidence below the
  diaphragm disappears with augmentation (5 of 15 to 0 of 12) and the frame and markers stay (6 of 15 to 5 of 12).
  The labels agree with the corner-and-bottom statistic (AUC 0.85) and weakly with the background one (0.65).
- Source: `results/shortcut/summary.md`; each result against the reading fixed before it was computed in
  NOTES.md, the #15 Decisions row.

## 6. What we learned about the method

- **The validation set is at its ceiling.** ROC-AUC 0.994-1.000 and specificity 0.955-1.000 in every run, so it
  can barely tell the runs apart (`runs.csv`). During tuning this moved checkpoint selection from ROC-AUC to
  balanced accuracy (NOTES.md, training recipe). In the sweep, five checkpoints came from epoch 1 or 2, and a
  threshold chosen on validation gave no consistent advantage on test: better / worse / equal in 5 / 15 / 7
  experiments (seed 42) and 10 / 11 / 6 (seed 43) (#31).
- **Retraining is as noisy as many of the effects.** Seed 43 minus seed 42 moves test ROC-AUC by a median 0.006
  (90th percentile 0.023) and the stability gap by 0.047 - outside a single run's interval in 11 and 23 of the 27
  experiments. A single seed's intervals would have overstated several findings; hence the two-seed rule (#32).
- **Invisible *added* noise flips predictions to pneumonia; the dataset's own texture points the other way.**
  Noise too faint to see flips 5,272 test predictions from NORMAL to PNEUMONIA and 28 the other way (seed 42), in
  every architecture and regime, and it flipped validation predictions at similar rates during the calibration
  (#7). The errors lean the same way: 2,267 false alarms against 90 misses at 0.5. The healthy images are the
  grainier ones in the data, so this is not a texture the models learned (section 5).
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

## 7. Limitations the report should state

- One dataset. The test set comes from the same sources, and the two classes differ in image size, aspect
  ratio, framing and attached devices (#15); squashing to 224x224 cannot remove the aspect-ratio difference.
- Patient grouping comes from filenames: the NORMAL groups are case keys, not proven patients, and `person<N>`
  merges children who are probably distinct (NOTES.md, the patient-key row).
- Two seeds per configuration; the intervals cover test sampling, not training.
- The validation set is at its ceiling and does not show the test set's shift (sections 4 and 6).
- The shortcut check is partly correlational: the size finding holds on test only and size was never changed in
  a test; the encoding test copies the quality table only; the visual audit is one rater, not blind, on 6 runs of
  one seed (section 5).
- Stability is measured with one perturbation per family - a small rotation in-family, faint noise
  out-of-family - fixed before the sweeps and kept (#29); blur as a second out-of-family perturbation is future
  work. The metric changed from the proposal's cosine similarity to Pearson correlation - the report has to say so
  and why (#30).
- Class weighting and the validation-chosen threshold correct the training imbalance, not the test set's
  operating point (#17, #27).

## 8. Still open

- `experiments/experiment_log.md` - the results rows (Or).
- #5 - whether `README.md` links the block diagram (Or).
- The report states the change from cosine to Pearson (#30).
