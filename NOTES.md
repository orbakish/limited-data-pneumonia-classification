# Project Notes

A shared running log of open questions, decisions, and ideas — so nothing gets lost between
work sessions or between us.

**How to use this file:**
- **Open questions** — anything that needs a decision or an answer from the other person.
  Note who it's for and its status.
- **Decisions made** — once resolved, move it here *with the reasoning*, so we don't
  re-litigate it later or forget why we chose it.
- **Parking lot** — ideas worth remembering but deliberately not doing now.

Add to it freely as things come up rather than waiting until the end of a session.

---

## Open questions

| # | Question | For | Status |
|---|---|---|---|
| 1 | Should the 20% / 50% training subsets **preserve class balance**, or be sampled uniformly at random? The dataset is known to be imbalanced (more pneumonia than normal), so this affects how comparable the limited-data results are. Depends on the class counts from the data exploration. | Or → Yotam | Open |
| 2 | `notebooks/data_exploration.ipynb` is currently a 0-byte file — it can't be opened in Jupyter as-is. Is the exploration work in progress elsewhere, or does the file need recreating? | Or | Open |
| 3 | `requirements.txt` currently lists 9 packages with **no version pins**, which makes results non-reproducible if a package updates mid-project. Proposal: pin all versions. Needs agreement since it affects both our setups. | Both | Open |
| 4 | `README.md` ends with a leftover `## Yotam first edit` heading from an early git practice commit — it has no content and should probably be removed. | Or (README is hers) | Open |
| 5 | Once `docs/architecture.md` (block diagram) exists, should it be linked from `README.md` so it's discoverable? | Or (README is hers) | Open |
| 6 | Freeze the pretrained conv layers vs. fine-tune them? Kermany et al. found full fine-tuning *hurt* performance on limited medical data due to overfitting. Leaning toward the frozen-backbone recipe as the default. | Yotam, to confirm | Open |
| 7 | What exactly counts as the "small input perturbation" for the Grad-CAM stability metric? Needs a precise, written definition before any stability numbers are produced. | Yotam | Open |

## Decisions made

| Decision | Reasoning | Date |
|---|---|---|
| **Training runs on Google Colab Pro**; local machine is for development and small smoke tests only. | 27 experiments × 3 architectures is too much for a local CPU-only setup. | 2026-09-08 |
| **Nothing important lives on the Colab session disk** — dataset, checkpoints, and results all live in Google Drive. | Colab sessions time out and their filesystem is ephemeral; a disconnect would otherwise destroy completed work. | 2026-09-08 |
| **Two entry points instead of one macro script**: `run_experiment.py` runs a single config by experiment ID; `run_all.py` loops over all 27 calling that same path. | Colab sessions drop. A single monolithic "run everything" script means one disconnect at experiment 19 costs the whole sweep. This way individual experiments can be re-run or resumed. | 2026-09-08 |
| **Grad-CAM and the stability metric are implemented *before* the full 27-config sweep**, and validated on a single-config pilot run (E01) end-to-end. | Grad-CAM needs trained weights. If the sweep ran first and checkpoints turned out to be saved in an unusable form, all 27 runs would need repeating. The pilot proves the whole pipeline before GPU hours are committed. | 2026-09-08 |
| **Reference papers stored in `report/papers/`** (PDFs committed) with written summaries in `report/papers/summaries.md`. | Keeps the source material next to the report material, matching where `project_proposal.pdf` already lives. | 2026-09-08 |
| **Block diagram lives in `docs/architecture.md`**, written in Mermaid. | Mermaid renders natively on GitHub and is diffable as text, unlike an exported image. Kept as a separate file rather than a README section since README is Or's to maintain. | 2026-09-08 |

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
