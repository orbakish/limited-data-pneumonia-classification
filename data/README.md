# Dataset Setup

This project uses the Kaggle **Chest X-Ray Images (Pneumonia)** dataset.

The dataset contains chest X-ray images from two classes:

* `NORMAL`
* `PNEUMONIA`

The dataset files are not stored in this GitHub repository because of their size.

## Expected Directory Structure

After downloading and extracting the dataset, the local directory should have the following structure:

```text
data/
└── raw/
    └── chest_xray/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        ├── val/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        └── test/
            ├── NORMAL/
            └── PNEUMONIA/
```

## Data Splits

The dataset ships with `train/`, `val/`, and `test/` folders, but the official `val/` split is very
small (on the order of 16 images in total). That is not enough to support early stopping or model
selection without very noisy results, and the same noise would affect all 27 experiments.

We therefore build our own validation split:

* The official `train/` and `val/` images are combined into a single pool.
* A fixed validation set of roughly 15% is carved out of that pool, stratified by class so the
  NORMAL / PNEUMONIA ratio is preserved.
* The remaining ~85% becomes the **training pool** that the experimental subsets are drawn from.
* The official `test/` split is left completely untouched and is never merged into the pool.

The split is made once with a fixed random seed, and the resulting file lists are written out, so
it is identical for every experiment.

### Patient-level splitting

The dataset contains multiple X-rays of the same patient, and both classes encode a patient (or
study) identifier in the filename.

Images from one patient must not appear in both the training and validation sets. If they do, the
model can key on patient-specific features such as body size, positioning, or imaging hardware
rather than on pathology, and validation scores become optimistically biased. Whole patients are
therefore assigned to one side of the split.

Patient keys, as established by `survey_dataset.py`:

| Class | Filename pattern | Patient key |
|---|---|---|
| PNEUMONIA | `person<N>_<type>_<M>` (optionally `_<K>`), where type is `bacteria` or `virus` | `person<N>_<type>` |
| NORMAL | `IM-<A>-<B>` (optionally `-<C>`), sometimes prefixed `NORMAL2-` | `[NORMAL2-]IM-<A>` |

**The infection type must be part of the PNEUMONIA key.** Bacterial and viral cases are numbered in
separate sequences, so `person53_bacteria_*` and `person53_virus_*` are different children. Keying on
`person<N>` alone wrongly merges them: 809 person numbers carry both a bacterial and a viral case
within a single split.

For NORMAL, filenames cannot tell a patient apart from a single imaging study, but grouping by `<A>`
still keeps all images from the same study together.

Using these keys, no patient appears in more than one of the official `train/`, `val/`, and `test/`
splits, so the official test set is patient-independent and is used as-is.

## Experimental Subsets

The project uses three training-data sizes, drawn from the training pool described above:

* 20% of the training pool
* 50% of the training pool
* 100% of the training pool

**"100%" means the whole training pool (roughly 4,450 images), not the whole original `train/`
folder.** This should be stated explicitly in the report so the numbers are not misread.

The subsets are **nested** - the 20% subset sits inside the 50% subset, which sits inside the full
pool. This is achieved by shuffling each class pool once with a fixed seed and taking prefixes of
that single ordering:

```text
normal_shuffled    = shuffle(NORMAL images,    seed)
pneumonia_shuffled = shuffle(PNEUMONIA images, seed)

100%  = all images in the training pool
50%   = normal_shuffled[:50%] + pneumonia_shuffled[:50%]
20%   = normal_shuffled[:20%] + pneumonia_shuffled[:20%]
```

Nesting matters because it means a change in results between data sizes comes from *adding* data,
not from drawing a different sample. Taking prefixes of one shuffled ordering guarantees this by
construction, rather than relying on a two-step calculation that could break it.

Sampling is done per class so the original class ratio is preserved at every data size. The
experiment is about data *quantity*, so the class balance must not vary along with it.

The combined subset list is not shuffled again - the DataLoader shuffles every epoch anyway.

The validation and test sets stay fixed across all experiments. Only the training data is
subsampled.

## Important

Do not commit the dataset images to GitHub.

Only code, experiment definitions, metrics, figures, and documentation should be stored in the repository.
