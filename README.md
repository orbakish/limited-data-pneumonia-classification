# Limited-Data Pneumonia Classification

Deep Learning course project — Evaluating Augmentation Reliability in Limited-Data Chest X-Ray Pneumonia Detection Using Grad-CAM.

## Project Overview

This project investigates automatic pneumonia detection from chest X-ray images under limited-data conditions.

The task is a binary image classification problem with two classes:

* Normal
* Pneumonia

The main goal is to study how training-data size and data augmentation strength affect both classification performance and the stability of model explanations.

## Dataset

We use the publicly available Kaggle **Chest X-Ray Images (Pneumonia)** dataset.

The dataset provides official training, validation, and test splits. For this project, the original training and validation sets are combined and a new validation split is created. The official test set remains untouched.

Dataset files are not stored in this repository due to size limitations.

Expected local structure:

```text
data/raw/
├── train/
├── val/
└── test/
```

## Models

The project uses transfer learning with three CNN architectures:

* MobileNetV3-Small
* ResNet18
* EfficientNet-B0

## Experimental Design

We compare three training-data sizes:

* 20%
* 50%
* 100%

For each data size and model, we evaluate three augmentation regimes:

* No augmentation
* Mild medically realistic augmentation
* Stronger generic augmentation

This results in 27 main training configurations.

All 27 experiments use partial fine-tuning and class-weighted cross-entropy, following the training recipe selected during the validation-only tuning phase.

## Evaluation

Classification performance will be evaluated using:


* Accuracy
* Precision
* Recall
* Specificity
* F1-score
* Balanced accuracy
* ROC-AUC
* Confusion matrices
* 95% confidence intervals

For threshold-dependent metrics, we report both the standard threshold of 0.5 and a threshold selected on the validation set to maximize balanced accuracy.

In addition, Grad-CAM is used to analyze model explanations and possible shortcut learning.

Explanation stability is measured on a fixed class-balanced subset of test images using two small perturbation families: an in-family rotation and out-of-family Gaussian noise. Stability is reported separately for both perturbations.

## Project Category

Healthcare, Computer Vision

## Documentation

See [`docs/architecture.md`](docs/architecture.md) for the project pipeline and architecture overview.

