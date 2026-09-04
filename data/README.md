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

## Experimental Subsets

The project will use three training-data sizes:

* 20% of the training set
* 50% of the training set
* 100% of the training set

The validation and test sets should remain fixed across experiments.

The 20% and 50% subsets should be created reproducibly using a fixed random seed so that all models and augmentation settings are compared using the same images.

## Important

Do not commit the dataset images to GitHub.

Only code, experiment definitions, metrics, figures, and documentation should be stored in the repository.
