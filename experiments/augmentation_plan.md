# Data Augmentation Plan

The project compares three augmentation regimes: no augmentation, mild medically realistic augmentation, and stronger generic augmentation.

The same augmentation definitions should be used for all three model architectures and all training-data sizes.

## 1. No Augmentation

Only the preprocessing required by the pretrained model will be applied.

Planned operations:

* Resize images to the model input size
* Convert to tensor
* Normalize using ImageNet normalization

No random transformations will be applied.

## 2. Mild Augmentation

The mild regime is intended to simulate small realistic variations in chest X-ray acquisition without significantly changing the image content.

Initial planned transformations:

* Small random rotation: approximately ±5°
* Small translation: up to approximately 2%
* Small scale variation: approximately 0.95–1.05
* Mild brightness/contrast variation

The purpose of this regime is to introduce limited variation while preserving the main anatomical structure.

## 3. Strong Augmentation

The strong regime will use larger transformations to test whether more aggressive augmentation improves classification performance or reduces explanation stability.

Initial planned transformations:

* Random rotation: approximately ±15°
* Translation: up to approximately 5%
* Scale variation: approximately 0.90–1.10
* Stronger brightness/contrast variation
* Random resized crop, while preserving most of the image

## Important Experimental Rules

* Augmentation is applied only to the training set.
* Validation and test images must not receive random augmentation.
* The same augmentation settings should be used for MobileNetV3-Small, ResNet18, and EfficientNet-B0.
* The same definitions should be used for the 20%, 50%, and 100% training-data experiments.
* Random seeds should be controlled where possible to improve reproducibility.

## Implementation Note

These values are the initial experimental definitions and may be slightly adjusted during implementation if necessary. Any changes should be documented here and kept fixed for all corresponding experiments.
