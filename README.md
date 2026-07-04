# limited-data-pneumonia-classification
Deep Learning course project - Pneumonia Detection using Transfer Learning under Limited Training Data

## Project Overview

This project investigates whether data augmentation can reduce the performance loss caused by limited labeled training data in chest X-ray pneumonia classification.

The model is based on transfer learning using ResNet18 in PyTorch. The planned experiments compare performance with and without data augmentation under different training data sizes: 20%, 50%, and 100%.

## Project Category

Healthcare, Computer Vision

## Dataset

The project uses a publicly available chest X-ray pneumonia dataset, such as the Kaggle Chest X-Ray Images (Pneumonia) dataset.

The dataset is not included in this repository due to size limitations.

Expected local structure:

```text
data/raw/
├── train/
├── val/
└── test/
