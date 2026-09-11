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
| 7 | What exactly counts as the "small input perturbation" for the Grad-CAM stability metric? Needs a precise, written definition before any stability numbers are produced. | Yotam | Open |
| 10 | Do the NORMAL filenames (`IM-0115-0001.jpeg`, `NORMAL2-IM-...`) encode a patient ID the way the PNEUMONIA ones do (`person53_bacteria_255.jpeg`)? This decides whether patient-level splitting can be applied to both classes or only to PNEUMONIA. If only PNEUMONIA, the limitation goes in the report. | Or (from EDA) | Open |

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
| **`torch`, `torchvision` and `numpy` are deliberately left UNPINNED** in `requirements.txt`; the analysis stack (pandas, matplotlib, scikit-learn, pillow, tqdm) is pinned to Colab's current versions. | Colab ships CUDA-matched builds of these. Pinning a version makes pip replace Colab's build, which can break GPU support. torch is compiled against a specific numpy major version, so numpy carries the same constraint. Or applies the same versions in her environment. | 2026-09-11 |
| **Reproducibility for torch/torchvision/numpy comes from recording versions per run**, not from pinning. | Those three determine the actual numbers, but they are exactly the three that cannot be safely pinned on Colab, so each run's recorded metadata has to carry them instead. | 2026-09-11 |
| **Local development stays on Python 3.12**, even though Colab runs 3.13. | Every pinned package has wheels for both versions, and local is used only for development and smoke tests. Every reported result comes from Colab, so matching the minor version is not worth the setup detour. | 2026-09-11 |
| **Training runs on an L4 or T4 GPU, whichever is available - not the A100.** | The workload is three small architectures on roughly 5,200 images at 224x224, which an L4 or T4 handles comfortably. A100s consume Colab Pro compute units several times faster, and the budget has to cover 27 runs plus any re-runs. | 2026-09-11 |
| **Limited-data subsets are nested prefixes of a single seeded shuffle, stratified by class.** Each class pool (NORMAL, PNEUMONIA) is shuffled once with a fixed seed; the 50% and 20% subsets are the first 50% and first 20% of that same ordering, then combined. The combined list is not shuffled again, and the resulting file lists are written out so nesting can be asserted in a test. | Nesting (20% inside 50% inside 100%) means a change between data sizes comes from *adding* data rather than from drawing a different sample, which removes a confound. Taking prefixes of one shuffle guarantees that nesting by construction, instead of depending on two-step arithmetic that could break it. Sampling per class preserves the original class ratio at every size, so the experiment isolates data quantity rather than class balance. A final shuffle is unnecessary because the DataLoader shuffles every epoch, and it would add a second seed to track. | 2026-09-11 |
| **We build our own validation split rather than using the official one.** The official `train/` and `val/` images are combined, a fixed validation set of roughly 15% is carved out stratified by class, and the training subsets are drawn from the remaining ~85%. The official `test/` split is left completely untouched. The split is made once with a fixed seed and its file lists are written out. Splitting is done at patient level wherever filenames allow it. | The official validation split is only around 16 images, far too small for early stopping or model selection without severe noise, and that noise would affect all 27 experiments. Test is left alone so results stay comparable to Kermany et al. and remain credible. Patient-level splitting prevents the same patient appearing on both sides, which would let the model key on patient-specific features rather than pathology and inflate validation scores. | 2026-09-11 |
| **"100%" means 100% of our post-split training pool (roughly 4,450 images), not 100% of the original Kaggle `train/` folder.** | Carving out our own validation set shrinks the pool the subsets are drawn from. If this is not stated explicitly, the reported data sizes are misleading. At this scale the 20% condition lands near 890 images, comparable to the 1,000-per-class limited model in Kermany et al. | 2026-09-11 |

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
