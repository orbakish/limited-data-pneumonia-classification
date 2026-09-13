# Data Augmentation Plan

The project compares three augmentation regimes: no augmentation, mild medically realistic augmentation, and stronger conventional augmentation.

The same augmentation definitions should be used for all three model architectures and all training-data sizes.

## Clinical Realism Constraint

All regimes are bound by one rule: **no transformation may change anatomy, left/right orientation, or the visibility of pathology.** Augmentation exists to reduce the amount of data needed for good classification, and a transformation that alters what the label describes introduces label noise instead.

Allowed transformation types:

* Rotation
* Translation
* Zoom (scale)
* Brightness
* Contrast

Excluded, in every regime:

* **Horizontal flip** — reverses laterality: the heart appears on the wrong side and L/R markers are mirrored
* **Random crop / random resized crop** — can remove the region containing the pathology
* **Random erasing / cutout** — can erase the pathology the label refers to
* **Elastic or non-rigid warps** — distort anatomical structure
* **Noise and blur** — not because they are unrealistic, but because they are reserved as the out-of-family perturbation for the Grad-CAM stability metric; training on them would invalidate that measurement

**Zoom is asymmetric.** Zooming out only adds border, so it can use a wider range. Zooming in is effectively a crop and can push pathology out of frame, so it stays tight.

## Preprocessing (not augmentation)

These steps are deterministic: every image is changed in exactly the same way, with nothing random. They are applied to all splits, including validation and test, because the models can only accept images in this format.

* Convert to grayscale, then replicate to 3 channels (the pretrained models expect 3 channels; this also removes the RGB/grayscale difference between classes)
* Resize directly to 224×224 (aspect ratio is not preserved; padding would add class-correlated black borders, and centre-cropping could remove lung fields)
* Convert to tensor
* Normalize using ImageNet normalization

**Augmentation is different:** it adds random variation and is applied to the training set only. Validation and test images receive preprocessing and nothing else.

## 1. No Augmentation

Training images receive only the preprocessing above. No random transformations.

This regime is the baseline for each architecture and data size.

## 2. Mild Augmentation

The mild regime simulates small realistic variations in chest X-ray acquisition without significantly changing the image content.

Candidate transformations:

* Random rotation: ±5°
* Translation: to be fixed from the dataset survey (see NOTES.md question #11)
* Zoom: to be fixed from the dataset survey (see NOTES.md question #11); zoom-in kept tight
* Brightness variation: ±10%
* Contrast variation: ±10%

The purpose of this regime is to introduce limited variation while preserving the main anatomical structure.

## 3. Strong Augmentation

The strong regime is a stronger conventional augmentation regime: the same allowed transformation types at larger magnitudes, with every transformation and magnitude constrained by the clinical realism rule above. It tests whether more aggressive augmentation improves classification performance or reduces explanation stability.

Candidate transformations:

* Random rotation: ±20°
* Translation: to be fixed from the dataset survey (see NOTES.md question #11); larger than mild
* Zoom: to be fixed from the dataset survey (see NOTES.md question #11); wider zoom-out than mild, zoom-in still tight
* Brightness variation: ±25%
* Contrast variation: ±25%

The random resized crop listed in the initial version of this plan has been removed, since it can crop pathology out of the image.

## Candidate Magnitudes Summary

| Transformation | No augmentation | Mild | Strong |
|---|---|---|---|
| Rotation | — | ±5° | ±20° |
| Translation | — | pending (#11) | pending (#11) |
| Zoom out | — | pending (#11) | pending (#11), wider than mild |
| Zoom in | — | pending (#11), tight | pending (#11), tight |
| Brightness | — | ±10% | ±25% |
| Contrast | — | ±10% | ±25% |

These are candidate values for initial testing, not final settings.

## Important Experimental Rules

* Augmentation is applied only to the training set.
* Validation and test images must not receive random augmentation — only the deterministic preprocessing.
* The same augmentation settings should be used for MobileNetV3-Small, ResNet18, and EfficientNet-B0.
* The same definitions should be used for the 20%, 50%, and 100% training-data experiments.
* Noise and blur are never used as augmentations (reserved for the stability metric).
* Random seeds should be controlled, including DataLoader worker seeds, to improve reproducibility.

## Implementation Note

The magnitudes above are candidates for initial testing and may be tuned after the single-configuration pilot run (E01), for example if mild and strong turn out to be barely distinguishable. Any change should be documented here and in NOTES.md, and then kept fixed for all corresponding experiments.

Revised 2026-09-13 to apply the clinical realism constraint (NOTES.md question #12, agreed by Or), remove the random resized crop, and record candidate magnitudes. Translation and zoom limits remain pending the dataset survey.
