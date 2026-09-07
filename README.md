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

The dataset is already divided into training, validation, and test sets. Dataset files are not stored in this repository due to size limitations.

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

## Evaluation

Classification performance will be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrices

In addition, Grad-CAM will be used to analyze model explanations.

Explanation stability will be measured on a fixed class-balanced subset of test images by comparing normalized Grad-CAM heatmaps before and after small input perturbations using cosine similarity.

## Project Category

Healthcare, Computer Vision

## Yotam first edit 
