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

### Patient/group-level splitting

The dataset may contain multiple X-rays from the same patient or imaging case, so related images
should not be split between training and validation. Otherwise, the model may partially recognize
patient- or acquisition-specific features instead of learning pathology-related features.

Patient/group keys were investigated during the dataset EDA.

| Class | Filename pattern | Grouping rule |
|---|---|---|
| PNEUMONIA | `person<N>_<type>_<M>` | `person<N>` |
| NORMAL | `IM-<A>-<B>` or `NORMAL2-IM-<A>-<B>` | `[NORMAL2-]IM-<A>` case/group key |

For PNEUMONIA, `person<N>` is a reliable grouping key. Multiple images — and in some cases both
`bacteria` and `virus` filenames — can share the same `person<N>` identifier, so all images with
the same identifier are kept together.

For NORMAL, the filename allows us to extract a consistent case/group key, but the dataset does
not provide enough information to prove that this key corresponds to a unique physical patient.
We therefore treat it as a case-level grouping key and document this as a dataset limitation.

The official `train/` and `val/` folders share a consistent identifier namespace and are combined
before creating the new validation split.

The official `test/` split is kept completely untouched. PNEUMONIA numbering restarts in the test
set, so identical `person<N>` values across train and test cannot be assumed to represent the same
patient.

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

## Dataset Exploration

A full read-only exploratory analysis of the original dataset is stored under:

`data/eda/`

Main findings include:

- 5,856 valid JPEG images and no corrupted image files
- strong class imbalance in the training pool
- systematic differences in image dimensions and aspect ratios between the two classes
- 283 RGB images, all belonging to the PNEUMONIA class
- exact-duplicate groups within individual splits, but no hash-level leakage across splits
- several unusually small or heavily cropped PNEUMONIA images
- potential dataset shortcuts such as acquisition text, laterality markers, wires/electrodes, and different framing

The original images are kept unchanged. Channel normalization, resizing, augmentation, and other
preprocessing are performed by the data pipeline rather than by modifying the source dataset.


## Important

Do not commit the dataset images to GitHub.

Only code, experiment definitions, metrics, figures, and documentation should be stored in the repository.
