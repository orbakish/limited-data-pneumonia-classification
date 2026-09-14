# Project Notes

A shared running log of open questions, decisions, and ideas — so nothing gets lost between
work sessions or between us.

**How to use this file:**
- **Open questions** — anything that needs a decision or an answer from the other person.
  Note who it's for and its status.
- **Decisions made** — once resolved, move it here *with the reasoning*, so we don't
  re-litigate it later or forget why we chose it.
- **Parking lot** — ideas worth remembering but deliberately not doing now.

Question numbers are stable IDs: once a question is answered it moves to Decisions, and its
number is never reused. Gaps in the numbering are normal.

Add to it freely as things come up rather than waiting until the end of a session.

---

## Open questions

| # | Question | For | Status |
|---|---|---|---|
| 2 | `notebooks/data_exploration.ipynb` is currently a 0-byte file — it can't be opened in Jupyter as-is. Is the exploration work in progress elsewhere, or does the file need recreating? | Or | Open |
| 5 | Once `docs/architecture.md` (block diagram) exists, should it be linked from `README.md` so it's discoverable? | Or (README is hers) | Open |
| 6 | Freeze the pretrained conv layers vs. fine-tune them? Kermany et al. found full fine-tuning *hurt* performance on limited medical data due to overfitting. Leaning toward the frozen-backbone recipe as the default. | Yotam, to confirm | Open |
| 7 | Exact magnitudes for the two stability perturbations: the in-family rotation angle and the out-of-family Gaussian noise level. The families themselves are decided (see Decisions); the values are not. Both must be small enough that the diagnosis would not change. | Yotam | Open, families decided |
| 13 | `results/README.md` defines a single `gradcam_stability` field per experiment, but stability is now measured two ways (see Decisions). Proposed schema change: replace it with `gradcam_stability_in_family` and `gradcam_stability_out_of_family` (the gap can be derived). | Or (results/README.md is hers) | Open |
| 14 | Yotam is running `survey_dataset.py`, which measures class counts per split, image sizes and aspect ratios, colour modes, filename and patient-ID patterns (evidence for #10), border margins, and a contact sheet of sample images. That overlaps your data exploration. Proposal: your EDA reuses these numbers (they will land under Reference numbers) and focuses on the qualitative side and the report's dataset section. Does that split work, or is there part of it you would rather own? | Or | Open |
| 15 | **The two classes differ in ways unrelated to pathology** (see Reference numbers, Dataset): PNEUMONIA images are much smaller (median 1160x776 vs 1654x1323 for NORMAL), wider relative to their height (median aspect 1.49 vs 1.23), and all 283 RGB images in the dataset are PNEUMONIA. They are also **framed differently**: NORMAL images show dark background beside the body at the left/right edges far more often (78% / 65% of a 300-image sample, against 35% / 32% for PNEUMONIA), so pneumonia images are cropped tighter. The PNEUMONIA contact sheet also shows tubes, wires, ECG electrodes and a timestamp overlay far more often than the NORMAL one (an observation from a 16-image sample, not a measurement) - a well-documented shortcut, since sick children have devices attached. A model could learn these instead of the lungs, and because the test set comes from the same sources, accuracy alone would not reveal it. Preprocessing removes the colour difference but cannot remove the aspect-ratio difference. How prominent should this be in the report - a limitation paragraph, or also a dedicated Grad-CAM check for heatmaps concentrating on edges and proportions rather than lung fields? | Or + Yotam | Open |
| 16 | Some images are already cropped so tightly that the lung apices or bases are missing entirely (for example `person407_virus_811`, `person325_virus_661`, `IM-0022-0001`, shown on the extremes contact sheet that `survey_dataset.py` saves to `data/processed/survey/`). For these, any translation or zoom-in removes lung tissue. Proposal: keep them, since removing them changes the dataset relative to Kermany et al. Should we count how many there are and mention them in the report, or exclude them? | Or | Open |
| 17 | **Class imbalance handling.** Our training pool is about 2.9 pneumonia images per normal image. Options: a class-weighted loss, a weighted sampler that balances batches, or no correction (Kermany et al. used none). Also the decision threshold for calling an image pneumonia (default 0.5). Whatever is chosen must be identical across all 27 runs, and it changes what precision, recall and F1 mean, so the report needs to state it. | Yotam, to decide before training | Open |
| 18 | **Training hyperparameters and checkpoint selection**, fixed across all 27 runs: optimizer and learning rate (Kermany et al. used Adam at 0.001), batch size, epoch budget, early stopping on our own validation split, and which validation metric picks the saved checkpoint (for example validation loss vs. F1). Needed before the E01 pilot run. | Yotam, to decide before training | Open |
| 19 | **Dataset zip for Drive.** Once the dataset is organised, upload it to `pneumonia-project/data/chest_xray.zip` in the shared Drive folder. Three constraints, each of which stays invisible for a long time if missed: **(a)** image filenames must not change — patient keys are parsed out of them (`person<N>_<type>`, `[NORMAL2-]IM-<A>`) and the patient-level split depends on that; **(b)** the official `test/` split stays untouched and is never merged into train or val; **(c)** no `__MACOSX` folder and no nested `chest_xray/chest_xray` inside the zip. The archive should contain `chest_xray/train/...`, not the split folders loose at the top level. | Or | Open |
| 20 | **`data/dataset_manifest.json`** — a small committed file describing the final dataset, so `notebooks/colab_runner.ipynb` verifies the extracted data against an agreed description instead of counts hardcoded when `survey_dataset.py` happened to run. **Copy `data/dataset_manifest.template.json`, fill it in, and commit it under the real name**; the template lists every field and what the notebook does with it. Deliberately no total (the notebook sums the parts, so the two cannot contradict each other) and no checksum (a zip already stores a CRC per file, so a corrupt upload fails at extraction). Best written out by whatever script organises the dataset rather than filled in by hand. | Or | Open |

## Decisions made

| Decision | Reasoning | Date |
|---|---|---|
| **Training runs on Google Colab Pro**; local machine is for development and small smoke tests only. | 27 experiments × 3 architectures is too much for a local CPU-only setup. | 2026-09-08 |
| **Nothing important lives on the Colab session disk** — dataset, checkpoints, and results all live in Google Drive. | Colab sessions time out and their filesystem is ephemeral; a disconnect would otherwise destroy completed work. | 2026-09-08 |
| **Two entry points instead of one macro script**: `run_experiment.py` runs a single config by experiment ID; `run_all.py` loops over all 27 calling that same path. | Colab sessions drop. A single monolithic "run everything" script means one disconnect at experiment 19 costs the whole sweep. This way individual experiments can be re-run or resumed. | 2026-09-08 |
| **Grad-CAM and the stability metric are implemented *before* the full 27-config sweep**, and validated on a single-config pilot run (E01) end-to-end. | Grad-CAM needs trained weights. If the sweep ran first and checkpoints turned out to be saved in an unusable form, all 27 runs would need repeating. The pilot proves the whole pipeline before GPU hours are committed. | 2026-09-08 |
| **Reference papers stored in `report/papers/`** (PDFs committed) with written summaries in `report/papers/summaries.md`. | Keeps the source material next to the report material, matching where `project_proposal.pdf` already lives. | 2026-09-08 |
| **Block diagram lives in `docs/architecture.md`**, written in Mermaid. | Mermaid renders natively on GitHub and is diffable as text, unlike an exported image. Kept as a separate file rather than a README section since README is Or's to maintain. | 2026-09-08 |
| **Colab environment recorded** (checked 2026-09-11): Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, numpy 2.1.3, pandas 2.2.3, matplotlib 3.10.0, scikit-learn 1.6.1, pillow 11.3.0, tqdm 4.67.3. GPU allocated: A100-SXM4-40GB. | Baseline snapshot of the environment that produces the actual results. | 2026-09-11 |
| **`torch`, `torchvision` and `numpy` are deliberately left UNPINNED** in `requirements.txt`; the analysis stack (pandas, matplotlib, scikit-learn, pillow, tqdm) is pinned to Colab's current versions. | Colab ships CUDA-matched builds of these. Pinning a version makes pip replace Colab's build, which can break GPU support. torch is compiled against a specific numpy major version, so numpy carries the same constraint. Or applies the same versions in her environment. Practical consequence: install with plain `pip install -r requirements.txt` and never with `--upgrade`, which would force-upgrade the three unpinned packages and defeat the whole arrangement. Upgrading pip itself is unrelated and fine. | 2026-09-11 |
| **Reproducibility for torch/torchvision/numpy comes from recording versions per run**, not from pinning. | Those three determine the actual numbers, but they are exactly the three that cannot be safely pinned on Colab, so each run's recorded metadata has to carry them instead. | 2026-09-11 |
| **Local development stays on Python 3.12**, even though Colab runs 3.13. | Every pinned package has wheels for both versions, and local is used only for development and smoke tests. Every reported result comes from Colab, so matching the minor version is not worth the setup detour. | 2026-09-11 |
| **Where local and Colab library versions differ, Colab is the authority.** Anything that works locally must still be confirmed on Colab before it is trusted. | Leaving torch, torchvision and numpy unpinned means local pip takes the latest while Colab keeps its own build, so local is currently *newer* (torch 2.14.0+cpu vs 2.11.0+cu128). Code using a newer API would pass locally and fail on Colab, and every reported result comes from Colab. | 2026-09-11 |
| **Training runs on an L4 or T4 GPU, whichever is available - not the A100.** | The workload is three small architectures on roughly 5,200 images at 224x224, which an L4 or T4 handles comfortably. A100s consume Colab Pro compute units several times faster, and the budget has to cover 27 runs plus any re-runs. | 2026-09-11 |
| **Limited-data subsets are nested prefixes of a single seeded shuffle, stratified by class.** Each class pool (NORMAL, PNEUMONIA) is shuffled once with a fixed seed; the 50% and 20% subsets are the first 50% and first 20% of that same ordering, then combined. The combined list is not shuffled again, and the resulting file lists are written out so nesting can be asserted in a test. | Nesting (20% inside 50% inside 100%) means a change between data sizes comes from *adding* data rather than from drawing a different sample, which removes a confound. Taking prefixes of one shuffle guarantees that nesting by construction, instead of depending on two-step arithmetic that could break it. Sampling per class preserves the original class ratio at every size, so the experiment isolates data quantity rather than class balance. A final shuffle is unnecessary because the DataLoader shuffles every epoch, and it would add a second seed to track. | 2026-09-11 |
| **We build our own validation split rather than using the official one.** The official `train/` and `val/` images are combined, a fixed validation set of roughly 15% is carved out stratified by class, and the training subsets are drawn from the remaining ~85%. The official `test/` split is left completely untouched. The split is made once with a fixed seed and its file lists are written out. Splitting is done at patient level wherever filenames allow it. | The official validation split is only around 16 images, far too small for early stopping or model selection without severe noise, and that noise would affect all 27 experiments. Test is left alone so results stay comparable to Kermany et al. and remain credible. Patient-level splitting prevents the same patient appearing on both sides, which would let the model key on patient-specific features rather than pathology and inflate validation scores. | 2026-09-11 |
| **"100%" means 100% of our post-split training pool (roughly 4,450 images), not 100% of the original Kaggle `train/` folder.** | Carving out our own validation set shrinks the pool the subsets are drawn from. If this is not stated explicitly, the reported data sizes are misleading. At this scale the 20% condition lands near 890 images, comparable to the 1,000-per-class limited model in Kermany et al. | 2026-09-11 |
| **Augmentation must stay clinically realistic: no transform may change anatomy, left/right orientation, or the visibility of pathology.** Allowed: rotation, translation, zoom, brightness, contrast. Excluded: horizontal flip, random crop, random erasing or cutout, elastic warps. **Framing (agreed with Or, question #12):** mild adds small realistic variation; strong remains a stronger conventional augmentation regime, with every transformation and magnitude constrained to preserve clinical realism. | The project aims to be a realistic clinical tool, and augmentation exists to reduce the data needed for good classification. A horizontal flip reverses laterality (the heart appears on the wrong side, L/R markers mirror). Crops, erasing and warps can remove or distort the very pathology the label describes, which is label noise. Or agreed to the realism constraint and kept the proposal's "stronger conventional/generic augmentation" label for strong, so the proposal framing stays intact while the constraint applies to both regimes. | 2026-09-13 |
| **Candidate magnitudes for initial testing** (provisional until the augmentation preview checks them on tightly framed images): rotation mild +-5 degrees, strong +-15 degrees; translation mild up to 2%, strong up to 5%; zoom-out mild to 0.95, strong to 0.85; zoom-in capped at 1.05 in both regimes; brightness and contrast mild +-10%, strong +-25%. Resolves question #11. | Strong has to be clearly separated from mild, or a three-level comparison risks a null result. The contact sheets show the lung-to-edge margin varies a lot and is tight in many PNEUMONIA images - lateral lung edges about 5-10% from the frame, lung bases near the bottom - so translation stays inside a 5% inset. Zoom-out only adds border; zoom-in is effectively a crop, so 1.05 (about 2.5% per side) is the cap. Strong rotation was reduced from +-20 to +-15 degrees because rotating a rectangular image swings its corners out of frame, and the costophrenic angles sit in the bottom corners. | 2026-09-13 |
| **Grad-CAM stability is measured two ways.** *In-family:* a small rotation, with the perturbed image's heatmap rotated back into alignment and the out-of-frame corners masked before comparing. *Out-of-family:* small Gaussian noise. Both are reported, along with the gap between them. | Augmentation trains a model to ignore specific changes, so a perturbation from that same family makes augmented models look more stable by construction - the metric would partly measure overlap between training augmentation and test perturbation, and would tilt the result against the proposal's own hypothesis that strong augmentation may reduce stability. Noise is outside every regime and moves no pixels, so it is the fair comparison. The in-family measurement is kept because the gap shows how much apparent stability is trained-in invariance. Rotated heatmaps need re-alignment, or even a perfectly consistent model would score low because the lungs moved. | 2026-09-13 |
| **Noise and blur are never used as augmentations.** | They are reserved as the out-of-family stability perturbation. Training on them would turn that measurement into another in-family one. | 2026-09-13 |
| **Dataset layout after extraction: keep only `data/raw/chest_xray/{train,val,test}/`.** The Kaggle zip also contains a nested duplicate (`chest_xray/chest_xray/`) and a `__MACOSX/` folder; both are deleted. Code lists image files non-recursively and skips hidden files. | The nested copy is byte-for-byte identical (same counts in every folder, same filenames). `__MACOSX/` holds 5,856 files named like `._IM-0115-0001.jpeg` - they match `*.jpeg` but are 268-byte macOS metadata, not X-rays. A recursive `*.jpeg` search over the extracted zip would find three times the real dataset, a third of it unreadable. Anyone downloading the zip should clean it the same way. | 2026-09-13 |
| **Every image is converted to grayscale (`L`) and then replicated to 3 channels**, regardless of its stored mode. | The ImageNet-pretrained models need 3 channels. All 283 RGB images are PNEUMONIA, so feeding native RGB could leave small channel differences or compression artefacts that correlate with the label. Routing every image through the same grayscale path erases the mode difference entirely. | 2026-09-13 |
| **Images are resized by squashing directly to 224x224** (aspect ratio not preserved). | No resize strategy removes the aspect-ratio difference between the classes, since it is in the data; each only changes how it leaks. Padding to square would leave roughly a third of each PNEUMONIA frame as black bars against about a fifth for NORMAL - a trivially learnable shortcut that also wastes resolution. Centre-cropping is excluded because it can remove lung fields. Squashing keeps all anatomy, adds no artificial borders, and leaves the subtlest form of the leak, which is documented (question #15) rather than hidden. | 2026-09-13 |
| **Patient keys for patient-level splitting:** PNEUMONIA `person<N>_<type>` (number *and* infection type), NORMAL `[NORMAL2-]IM-<A>`. Answers question #10: NORMAL filenames do group images, so patient-level splitting applies to both classes. | NORMAL `<A>` groups multiple images (75 groups of 2-4 images). Bacterial and viral cases are numbered in **separate sequences**, so `person<N>` alone is not a patient ID: 809 person numbers carry both a bacterial and a viral case inside the same split, which one child's image sequence cannot, and the 170 numbers shared between official train and test never once agree on infection type (chance agreement would be about 55%). Keying on the number alone had merged roughly 979 pairs of different children. Caveat: filenames cannot distinguish a patient from a single imaging study for NORMAL; grouping by `<A>` still keeps a study's images together. | 2026-09-13 |
| **The official `train/` / `test/` split is patient-independent**, so the official test set is used as-is with no patient-exclusion step. | Using the patient keys above, zero patients appear in more than one official split, for either class. This keeps results comparable to Kermany et al. The one blind spot is a child who had both a bacterial and a viral episode, who would carry two different keys - undetectable from filenames. | 2026-09-13 |
| **The Google Drive layout is `MyDrive/pneumonia-project/`**, holding `data/` (the dataset zip), `results/metrics/`, `results/figures/` and `checkpoints/`. The repo is cloned to `/content` and the dataset unzipped to `/content` fresh in every Colab session; nothing is cloned into Drive. | Drive is slow with many small files, so the dataset is stored as one zip and read in a single sequential pass. Only the zip, the results and the checkpoints have to survive a dropped session - the repo and the unpacked images are cheap to recreate. Keeping a git working copy on Drive is slow and prone to corrupting the index. | 2026-09-13 |
| **Colab installs with plain `pip install -r requirements.txt`, never `--upgrade`**, and `notebooks/colab_runner.ipynb` prints the torch and CUDA versions before and after the install and asserts they are unchanged. | torch, torchvision and numpy are unpinned on purpose, so pip treats them as already satisfied and leaves Colab's CUDA-matched builds alone. A pinned version or `--upgrade` replaces them with generic PyPI wheels and GPU support disappears with no error anywhere. The before/after assertion turns that silent failure into a loud one. | 2026-09-13 |
| **The repo stays private for the duration of the project**; whether to make it public is revisited after submission. | Consequence for both of us: running the Colab notebook needs a personal access token. It must be a **classic** token with the `repo` scope - fine-grained tokens only reach repositories owned by the account that issued them, and this repo is under Or's account. Store it as a Colab secret named `GITHUB_TOKEN` (the key icon in the Colab sidebar, with notebook access on). A token is never committed and never typed into a notebook cell. | 2026-09-13 |

## Reference numbers

Measured values worth citing in the report, recorded as they are established.

### Model sizes (torchvision pretrained, ImageNet weights)

| Model | Parameters |
|---|---|
| MobileNetV3-Small | 2,542,856 |
| EfficientNet-B0 | 5,288,548 |
| ResNet18 | 11,689,512 |

A 4.6x spread between the smallest and largest model, which gives the lightweight-vs-larger comparison three genuinely distinct size tiers rather than three similar models. Measured with `check_env.py` on 2026-09-11.

### Dataset (Kaggle Chest X-Ray Images, after cleanup)

Counts per official split:

| Split | NORMAL | PNEUMONIA | Total | Pneumonia : Normal |
|---|---|---|---|---|
| train | 1,341 | 3,875 | 5,216 | 2.89 |
| val | 8 | 8 | 16 | 1.00 |
| test | 234 | 390 | 624 | 1.67 |
| **all** | **1,583** | **4,273** | **5,856** | 2.70 |

Image properties per class, all splits combined:

| | NORMAL | PNEUMONIA |
|---|---|---|
| Width (min / median / max) | 912 / 1654 / 2916 | 384 / 1160 / 2772 |
| Height (min / median / max) | 496 / 1323 / 2713 | 127 / 776 / 2304 |
| Aspect w/h (min / median / max) | 0.88 / 1.23 / 2.11 | 0.84 / 1.49 / 3.38 |
| Colour modes | L: 1,583 | L: 3,990, RGB: 283 |

The size, aspect-ratio and colour-mode differences between classes are not explained by pathology and are a shortcut-learning risk (question #15). Outliers worth inspecting: a PNEUMONIA image only 127 px tall, and one with aspect ratio 3.38. Measured with `survey_dataset.py` on 2026-09-13.

Patients, from filenames (keys as defined in Decisions):

| | NORMAL | PNEUMONIA |
|---|---|---|
| Candidate patients | 1,444 | 2,653 |
| Patients with 1 image | 1,369 | 2,002 |
| Largest images-per-patient | 4 | 30 |
| Patients in more than one official split | 0 | 0 |

PNEUMONIA breakdown: 2,780 bacterial and 1,493 viral images. This matches Kermany et al. exactly (2,538 + 242 bacterial and 1,345 + 148 viral across their training and test sets), confirming this Kaggle copy is complete.

Dark background at image edges (seeded sample of 300 per class; fraction of image size):

| Edge | NORMAL median / p90 / share with any | PNEUMONIA median / p90 / share with any |
|---|---|---|
| Top | 0.000 / 0.000 / 1% | 0.000 / 0.000 / 0% |
| Bottom | 0.000 / 0.000 / 1% | 0.000 / 0.000 / 0% |
| Left | 0.024 / 0.077 / 78% | 0.000 / 0.053 / 35% |
| Right | 0.022 / 0.073 / 65% | 0.000 / 0.074 / 32% |

This measures dark background outside the body, not the margin around the lungs (the images have no scanner border). Its value is as framing evidence for question #15.
### Colab training environment (confirmed 2026-09-13)

| Item | Value |
|---|---|
| GPU | Tesla T4 |
| torch | 2.11.0+cu128 (CUDA 12.8) |
| `torch.cuda.is_available()` | True |

Confirmed end to end by `notebooks/colab_runner.ipynb` sections 1-8: the torch and CUDA versions were identical before and after `pip install -r requirements.txt`, so the pinned analysis packages installed without disturbing Colab's CUDA-matched deep-learning stack. This is the environment every reported result comes from.

## Parking lot

- **Guided Grad-CAM** (Grad-CAM × Guided Backpropagation, from Selvaraju et al. §3.2) — gives
  higher-resolution visualizations. Not in our project scope, but worth knowing the name in
  case it comes up in review as "why not Guided Grad-CAM?"
- **A polished, redrawn block diagram** for the final report and presentation. The Mermaid
  version is the working source of truth; a nicer figure can be made from it later.
- **Weighted error scoring** (Kermany et al. penalize false negatives more heavily than false
  positives, since a missed urgent case is worse than a false alarm). Not needed for our plain
  binary task, but a useful angle if the report wants to discuss clinical cost asymmetry.
- **Additional references** on data augmentation, transfer learning, and explanation stability
  — `report/references.md` already flags that these are still to be added.
