# Chest X-Ray Dataset — Exploratory Data Analysis (EDA) Report

**Scope:** READ-ONLY analysis. No file was deleted, moved, renamed, resized, or otherwise modified.
**Dataset root:** `chest_xray/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}`
**Total images analyzed:** 5,856 (all `.jpeg`)
**System files ignored:** `.DS_Store` (and any `._*`, `Thumbs.db`, `desktop.ini`).

Artifacts produced alongside this report:
- `image_inventory.csv` — one row per image (path, split, class, dims, mode, channels, patient_id candidate, size, sha256 hash).
- `patient_summary.csv` — patient/group candidate × class × split × image count.
- `anomalies.csv` — every flagged file + reason.
- `brightness.csv` — per-image mean/std brightness (used for near-black/white detection).
- `eda_figures/` — histograms, scatter, class bar, and visual-inspection montages.

---

## Executive Summary

- The dataset is clean at the file level: **0 non-image files** and **0 corrupted/unreadable images**; every file is `.jpeg`.
- There is a **clear systematic difference between classes in image size and aspect ratio**: NORMAL images are larger and closer to square (median AR ≈ 1.24), PNEUMONIA images are smaller and wider (median AR ≈ 1.49). This is a real **confounder** the model could exploit.
- **283 RGB images exist and ALL of them are PNEUMONIA (all in `train/PNEUMONIA`)**. The remaining 5,573 images are grayscale (`L`). Visually the RGB images are gray content stored as 3 channels. This mode↔class correlation is another potential shortcut.
- **Strong class imbalance** in train (PNEUMONIA:NORMAL ≈ 2.89:1). `val` is tiny (8 + 8 = 16 images) and balanced; `test` is 234/390 (≈1.67:1).
- **Patient IDs are recoverable for PNEUMONIA** (`personN`) and **train+val share one consistent numbering namespace**, so a patient-level split of the train+val pool is feasible for PNEUMONIA. **test is numbered independently** (person numbers restart at 1), so apparent train↔test person overlaps are numbering collisions, **not** proven same-patient leakage.
- For **NORMAL**, filenames follow `IM-####-####` and `NORMAL2-IM-####-####` patterns. A group key (the `IM-####` / `NORMAL2-IM-####` stem) can be extracted, but **it is NOT proven to be a unique patient ID** (evidence below shows two different `IM` numbers pointing to byte-identical images).
- **30 exact-duplicate hash groups (62 files)** exist. **All duplicates are within the same split and same class** → no train/val/test hash-level leakage and no cross-class duplication detected.
- Automatic anomaly detection flagged **426 rows** (mostly very wide/very small PNEUMONIA images). The **3 most extreme "cropped" images** are `person325_virus_659/660/661` (thin horizontal strips, AR ≈ 3.2, ~445×135).
- Visual inspection surfaced likely confounders: **laterality "R" markers, acquisition text, ECG electrodes/discs, tubes/wires, pediatric body-habitus, and tighter framing** — several of these are more prominent in PNEUMONIA.

---

## 1. Dataset Inventory

| split | class     | image_count | non_image_files | corrupted | total |
|-------|-----------|-------------|-----------------|-----------|-------|
| train | NORMAL    | 1341        | 0               | 0         | 1341  |
| train | PNEUMONIA | 3875        | 0               | 0         | 3875  |
| val   | NORMAL    | 8           | 0               | 0         | 8     |
| val   | PNEUMONIA | 8           | 0               | 0         | 8     |
| test  | NORMAL    | 234         | 0               | 0         | 234   |
| test  | PNEUMONIA | 390         | 0               | 0         | 390   |
| **all** |         | **5856**    | **0**           | **0**     | 5856  |

- **Extensions:** `.jpeg` only (5,856 / 5,856).
- **Non-image files:** none (after ignoring system files).
- **Corrupted / unreadable:** none. Every image opened and fully decoded (`PIL.Image.load()`), with `LOAD_TRUNCATED_IMAGES = False` during the check so truncation would have been caught.

---

## 2. Image Modes / Channels

Overall: **`L` (grayscale, 1 channel): 5,573** &nbsp;|&nbsp; **`RGB` (3 channels): 283**. No other modes.

| split | class     | L (grayscale) | RGB |
|-------|-----------|---------------|-----|
| train | NORMAL    | 1341          | 0   |
| train | PNEUMONIA | 3592          | 283 |
| val   | NORMAL    | 8             | 0   |
| val   | PNEUMONIA | 8             | 0   |
| test  | NORMAL    | 234           | 0   |
| test  | PNEUMONIA | 390           | 0   |

**RGB verification (as requested):**
- Total RGB = **283**.
- **100% of RGB images are PNEUMONIA**, and **all 283 are in `train/PNEUMONIA`**.
- Example filenames: `person1010_virus_1695.jpeg`, `person1035_bacteria_2969.jpeg`, `person1052_virus_1751.jpeg`, `person1073_bacteria_3008.jpeg`, …
- Visual check (`eda_figures/insp_rgb.png`): content is visually grayscale; RGB is just an encoding (3 channels). **However, "mode == RGB" perfectly predicts PNEUMONIA in this data → potential shortcut.** Always normalize channel count in preprocessing.

---

## 3. Image Dimensions & Aspect Ratio

`aspect_ratio = width / height`. Full per-split/class stats are in `eda_stats.txt`; aggregate class stats below.

| group | n | w min/med/max | h min/med/max | AR min/med/max | AR mean |
|-------|---|---------------|---------------|----------------|---------|
| **ALL NORMAL** | 1583 | 912 / **1654** / 2916 | 496 / **1323** / 2713 | 0.877 / **1.235** / 2.107 | 1.249 |
| **ALL PNEUMONIA** | 4273 | 384 / **1160** / 2772 | 127 / **776** / 2304 | 0.835 / **1.490** / 3.379 | 1.515 |

Aspect-ratio quantiles (5/25/50/75/95%):
- **NORMAL:** 1.023 / 1.145 / 1.235 / 1.336 / 1.519
- **PNEUMONIA:** 1.176 / 1.355 / 1.490 / 1.645 / 1.917

Width quantiles (5/25/50/75/95%):
- **NORMAL:** 1252 / 1472 / 1654 / 1848 / 2280
- **PNEUMONIA:** 817 / 1000 / 1160 / 1360 / 1710

Height quantiles (5/25/50/75/95%):
- **NORMAL:** 927 / 1140 / 1323 / 1554 / 2036
- **PNEUMONIA:** 480 / 640 / 776 / 960 / 1320

**Systematic difference — YES (clear).**
- NORMAL are **larger** (median 1654×1323) and **more square** (median AR ≈ 1.24).
- PNEUMONIA are **smaller** (median 1160×776) and **wider** (median AR ≈ 1.49).
- The distributions overlap but are clearly shifted; see `eda_figures/hist_width.png`, `hist_height.png`, `hist_aspect.png`, and `scatter_wh.png`.
- This pattern holds across splits (e.g. train/NORMAL AR median 1.223 vs train/PNEUMONIA 1.488).

**Implication:** size/AR alone carries class signal. If images are naively cropped/resized without care, the model can learn "smaller & wider ⇒ PNEUMONIA".

---

## 4. Patient ID Analysis

### PNEUMONIA — reliable
- Every PNEUMONIA filename matches `person<N>_...` (`bacteria`/`virus`). Extraction rule: **`patient_id = person<N>`**. (4,273 / 4,273 matched; 0 unknown.)
- **Unique person IDs (all splits): 1,674.**
- **Images per person:** min **1**, max **31**, median **2**, mean **2.55**.
- Subtypes: **bacteria = 2,780**, **virus = 1,493**.
- A single person can carry **both** subtypes and multiple views — e.g. `person1481` has `bacteria_3862…3868` **and** `virus_2567` (see `eda_figures/insp_person1481.png`). → confirms `personN` groups multiple images of one case.

**Cross-split person appearance:** 170 person IDs appear in **both `train` and `test`**. **This is almost certainly a numbering artifact, not real patient leakage:**
- `train` person numbers span **1 … 1945**; `val` continues at **1946 … 1954**; `test` **restarts at 1 … 1685**.
- Because `test` numbering overlaps the `train` range but is a separate sequence, `person1(train) ≠ person1(test)`.
- **UNCERTAIN:** we cannot prove from filenames whether any individual test person is physically the same as a train person. Since `test` stays untouched, this does not affect our train+val re-split. **Do not attempt to match test persons to train.**

### NORMAL — group key extractable, patient identity UNCERTAIN
Filename patterns (digits shown as `#`):

| count | pattern | example |
|-------|---------|---------|
| 858 | `NORMAL2-IM-#-#` | `NORMAL2-IM-0383-0001.jpeg` |
| 582 | `IM-#-#` | `IM-0115-0001.jpeg` |
| 84  | `IM-#-#-#` | `IM-0429-0001-0001.jpeg` |
| 59  | `NORMAL2-IM-#-#-#` | `NORMAL2-IM-0395-0001-0001.jpeg` |

- Two apparent acquisition sources: `IM-####` and `NORMAL2-IM-####`.
- Proposed group key: the stem before the first per-image counter → `IM-####` or `NORMAL2-IM-####` (e.g. `IM-0629`, `NORMAL2-IM-0416`).
- **Group sizes:** unique group candidates ≈ 1,444; images/group min **1**, median **1**, max **4** (e.g. `IM-0629`, `IM-0553`, `IM-0523` each have 4).
- The trailing `-0001`, `-0002` suffixes clearly denote multiple frames of the **same** case (byte-identical in several cases — see §5).

**Is the group key a true patient ID? UNCERTAIN / likely NO as a unique key.**
- Counter-evidence: `NORMAL2-IM-0095-0001.jpeg` and `NORMAL2-IM-0096-0001.jpeg` are **byte-identical** (same sha256) yet carry **different IM numbers**. So an `IM` number is **not** guaranteed unique per physical image/patient.
- Conclusion: `IM-####` / `NORMAL2-IM-####` is a reasonable **case/group** key to prevent obvious intra-case leakage, but **we cannot claim from filenames that it equals a unique patient**. Flag as **UNCERTAIN**.

`patient_summary.csv` lists every `(patient_id_candidate, class, split, num_images)`.

---

## 5. Duplicate / Leakage Check (sha256)

- **Duplicate hash groups: 30**, involving **62 files**.
- **Within same split: 30** — **Across splits (train/val/test): 0** — **Across classes: 0**.
- → **No hash-level leakage between train/val/test and no cross-class duplicates.**

Representative duplicate groups (byte-identical, see `eda_figures/insp_duplicates.png`):
- `train/PNEUMONIA`: `person258_bacteria_1207/1208/1209` (triple); `person1372_bacteria_3501/3502/3503` (triple); `person1481_bacteria_3862/3863`, `…3864/3865`, `…3866/3867`; `person124_virus_246/247`; `person1261_virus_2147/2148`; `person30_bacteria_156/157`; many more consecutive pairs.
- `train/NORMAL`: `NORMAL2-IM-0587-0001-0001 / -0002`.
- `test/NORMAL`: `NORMAL2-IM-0173-0001-0001 / -0002`, `NORMAL2-IM-0246-0001-0001 / -0002`, and notably **`NORMAL2-IM-0095-0001` = `NORMAL2-IM-0096-0001`** (different names, identical bytes).
- `test/PNEUMONIA`: `person1661_virus_2872/2873`, `person128_bacteria_606/607`, `person134_bacteria_643/644`.

Full list of duplicate members is in `image_inventory.csv` (group by the `hash` column).

**Implication:** duplicates are intra-split, so they don't leak across splits, but when we re-split train+val we should keep duplicate/same-case images together (patient/group-level split handles this).

---

## 6. Class Imbalance

| split | NORMAL | PNEUMONIA | ratio PNEU:NORMAL |
|-------|--------|-----------|-------------------|
| train | 1341 | 3875 | **2.89 : 1** |
| val | 8 | 8 | 1.00 : 1 |
| test | 234 | 390 | 1.67 : 1 |
| **train+val** | 1349 | 3883 | **2.88 : 1** |

**Meaning (informational only, no decision made):** with ~74% PNEUMONIA in the training pool, a naive model can reach high accuracy by over-predicting PNEUMONIA. Prefer class-aware metrics (recall/precision per class, F1, ROC-AUC/PR-AUC, balanced accuracy) over raw accuracy. Whether to address it via weighted loss, resampling, or nothing is left open per your instruction. Note the tiny `val` (16 images) is far too small to be a reliable validation set — consistent with your plan to build a new validation split from train+val.

---

## 7. Suspicious / Extreme Images

Automatic flags (see `anomalies.csv`, 426 rows). Reasons used:
- `very_tall_AR` (AR ≤ 0.60) / `very_wide_AR` (AR ≥ 1.90) — **239 wide flags**, 0 extremely tall.
- `small_*` (w<400 or h<400) and `tiny_*` (w<300 or h<300).
- `near_black_mean` / `near_white_mean` / `low_contrast_std` from the brightness pass (none crossed the black/white thresholds; contrast was reasonable across the set).

**The ~3 heavily "cropped" images (as you suspected):** the most extreme are all narrow horizontal strips showing only a thin band of the thorax:
- `person325_virus_659.jpeg` — 445×140, AR 3.18
- `person325_virus_660.jpeg` — 400×138, AR 2.90
- `person325_virus_661.jpeg` — 446×132, AR 3.38

(Closely followed by `person407_virus_811.jpeg` — 384×127, AR 3.02, and `person1684_bacteria_4462.jpeg` — 494×151, AR 3.27.) All are in `train/PNEUMONIA`. See `eda_figures/insp_smallest.png` and `insp_widest.png`.

**Visual notes (one line each):**
- `person325_virus_659/660/661` — three near-identical, severely cropped horizontal strips of a pediatric chest; only mid-thorax visible, top/bottom cut off.
- `person407_virus_811.jpeg` — smallest image overall (384×127); tightly cropped band, low detail.
- `person1684_bacteria_4462.jpeg` — very wide crop, ribs only, shoulders/diaphragm cut off.
- Smallest-area montage overall: uniformly pediatric, tightly cropped PNEUMONIA frames — a distinct sub-population.

There are **no near-black or near-white images**; the smallest by area is `person407_virus_811.jpeg` (384×127).

---

## 8. Possible Dataset Confounders (shortcuts)

From visual inspection of representative NORMAL and PNEUMONIA samples (`eda_figures/insp_normal_sample.png`, `insp_pneumonia_sample.png`, `insp_rgb.png`, `insp_duplicates.png`). We deliberately did **not** make any medical judgment; we only note non-disease cues a CNN might latch onto.

- **Laterality "R" marker:** present in the corner of the vast majority of images in **both** classes, but position/font/style varies. A burned-in letter is exactly the kind of high-contrast token a CNN can exploit. Examples: NORMAL `NORMAL2-IM-1322-0001` (top-left "R"); PNEUMONIA `person60_virus_117` (large "R" top-left).
- **Acquisition text / annotations:** some PNEUMONIA images have burned-in text (e.g. `person60_virus_117` "mA…"/exposure text top-right; `person358_virus_737` vertical text on the right edge).
- **ECG electrodes / round discs & wires/tubes:** visible mainly in PNEUMONIA, e.g. `person490_bacteria_2068` (disc on right shoulder), `person87_bacteria_434` (disc top-left), `person124_virus_246/247` (leads/wires), `person1261_virus_2147/2148` (curved wire across chest), `person1059_bacteria_2993` (line/tube top-left).
- **Body habitus / age:** NORMAL sample skews toward larger/older chests (shoulders, neck visible, full PA framing); PNEUMONIA skews **pediatric** (rounder shoulders, arms-up positioning, tighter framing). This correlates with the size/AR gap in §3.
- **Framing / cropping / borders:** NORMAL images typically full-field with black borders and more anatomy (neck→diaphragm); PNEUMONIA more zoomed onto the lungs. Black-bar borders vary.
- **Channel encoding:** RGB-only-in-PNEUMONIA (§2) is itself a confounder.

**Takeaway:** several cues (image size/AR, RGB encoding, R-marker/text placement, tubes/electrodes, pediatric framing) correlate with class for reasons unrelated to pathology. Mitigations to consider later: consistent center-crop/normalization, channel unification, and — if feasible — masking burned-in text/markers. (No changes made now.)

---

## 9. Preprocessing Implications (recommendations only — nothing changed)

- **Grayscale normalization:** data is fundamentally single-channel X-ray. Recommend converting everything to grayscale `L` first for a consistent starting point, then normalize (per-dataset mean/std, or simple [0,1]).
- **3-channel conversion for pretrained torchvision models:** ImageNet backbones expect 3 channels. Recommended path: **grayscale → replicate to 3 identical channels**, then apply ImageNet mean/std. This also **neutralizes the RGB-only-in-PNEUMONIA shortcut** (all images end up 3-channel identically). Do **not** feed the raw 283 RGB as-is while others are 1-channel.
- **Resize target:** 224×224 is standard for ResNet/DenseNet/EfficientNet-B0.
- **Direct resize to 224×224 — distortion risk: YES, non-trivial.** Median AR differs by class (NORMAL ≈ 1.24 vs PNEUMONIA ≈ 1.49) and the full AR range is ~0.84–3.38. A direct square resize stretches PNEUMONIA (and especially the wide crops) more than NORMAL, which **re-injects the class-correlated shape difference** and distorts anatomy.
- **Resize+padding vs direct resize:** **prefer aspect-preserving resize + padding** (letterbox to 224×224) or resize-shorter-side-then-center-crop, so a fixed geometry doesn't encode class. This reduces the AR confounder. Direct resize is simpler but bakes in distortion; if used, be aware AR is partly a label proxy.
- The extreme crops (§7, e.g. `person325_virus_659/660/661`) will look very degraded at 224²; consider handling/inspecting them explicitly.

---

## 10. Proposed Patient-Level Split Feasibility (no split created)

Based on filenames only, over the **train+val pool** (test stays untouched):

- **PNEUMONIA — feasible & reliable.**
  - Rule: `patient_id = person<N>` (regex `^person(\d+)_`).
  - train+val PNEUMONIA: **3,883 images across 1,642 unique persons** (median 2 imgs/person, mean 2.36, max 30).
  - train (1…1945) and val (1946…1954) share one numbering namespace with **no overlap**, so grouping by `personN` across the merged pool is safe.
- **NORMAL — partially feasible; patient identity UNCERTAIN.**
  - Rule (best available): group key = `IM-####` or `NORMAL2-IM-####` stem (strip trailing `-000x` frame counters).
  - train+val NORMAL: **1,349 images across ~1,219 group candidates** (median 1 img/group, mean 1.11, max 4).
  - **Caveat:** this key prevents intra-case leakage (e.g., `…-0001` / `…-0002` of the same case stay together), but it is **NOT proven to be a unique patient** — two different `IM` numbers can even be byte-identical (`NORMAL2-IM-0095-0001` = `NORMAL2-IM-0096-0001`). Treat it as a *case/group* key, not a guaranteed patient key.
- **What we still cannot know from filenames:**
  - Whether distinct `IM-####` groups belong to the same NORMAL patient (no patient token exists in NORMAL names).
  - Whether any `test` person equals any `train` person (independent numbering; and moot since test is frozen).
  - True patient identity behind PNEUMONIA `personN` beyond "same source case" (we assume person = patient, which is the dataset convention but unverifiable from names alone → **UNCERTAIN but the standard assumption**).

**Recommended approach when we build the new ~15% validation split (later):**
1. Merge original `train` + `val` into one pool (test frozen).
2. Group by: PNEUMONIA → `personN`; NORMAL → `IM-####`/`NORMAL2-IM-####` stem.
3. Split at the **group** level (GroupShuffle / StratifiedGroupKFold) stratified by class so no group crosses train/val.
4. Build nested **20% ⊂ 50% ⊂ 100%** subsets of the training pool at the **group** level, preserving class ratio.

---

## Anomalies Summary

- 426 flagged rows in `anomalies.csv` (dominant reasons: `very_wide_AR` 239; many `small_*`/`tiny_*`, all PNEUMONIA/train).
- 3 severe crops: `person325_virus_659/660/661`.
- 0 corrupted, 0 non-image, 0 near-black/near-white.

## Unresolved / UNCERTAIN Questions

1. **NORMAL patient identity** — `IM-####` is a case/group key, not a proven unique patient; different IM numbers can be identical images. **UNCERTAIN.**
2. **PNEUMONIA person = physical patient** — standard dataset assumption, unverifiable from filenames. **UNCERTAIN (assumed true).**
3. **train↔test person overlap** — almost certainly independent numbering (collisions), not leakage; cannot be proven either way. **UNCERTAIN, but irrelevant since test is frozen.**
4. Whether burned-in markers/text meaningfully bias training — flagged qualitatively; not quantified.

## Recommendations (summary)

- Unify channels (grayscale → 3-channel replicate) to kill the RGB↔PNEUMONIA shortcut.
- Use aspect-preserving resize + padding (letterbox) to 224×224 to reduce the AR confounder; avoid naive square resize if possible.
- Do the new validation split and the 20/50/100% subsets at **group level** (person / IM stem), stratified by class.
- Report class-aware metrics due to ~2.9:1 imbalance; the original `val` (16 imgs) is too small — rebuild it.
- Keep `test` untouched and do **not** cross-reference test person IDs with train.
- Be aware size/AR alone is partly predictive of the label; consider it when interpreting model behavior (e.g., Grad-CAM sanity checks on markers/borders).
