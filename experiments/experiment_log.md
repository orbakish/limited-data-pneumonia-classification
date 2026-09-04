# Experiment Log

This file tracks the main training configurations and their results.

| ID  | Model             | Data Size | Augmentation | Accuracy | Precision | Recall |  F1 | Grad-CAM Stability | Notes |
| --- | ----------------- | --------: | ------------ | -------: | --------: | -----: | --: | -----------------: | ----- |
| E01 | MobileNetV3-Small |       20% | None         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E02 | MobileNetV3-Small |       20% | Mild         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E03 | MobileNetV3-Small |       20% | Strong       |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E04 | MobileNetV3-Small |       50% | None         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E05 | MobileNetV3-Small |       50% | Mild         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E06 | MobileNetV3-Small |       50% | Strong       |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E07 | MobileNetV3-Small |      100% | None         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E08 | MobileNetV3-Small |      100% | Mild         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E09 | MobileNetV3-Small |      100% | Strong       |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E10 | ResNet18          |       20% | None         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E11 | ResNet18          |       20% | Mild         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E12 | ResNet18          |       20% | Strong       |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E13 | ResNet18          |       50% | None         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E14 | ResNet18          |       50% | Mild         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E15 | ResNet18          |       50% | Strong       |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E16 | ResNet18          |      100% | None         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E17 | ResNet18          |      100% | Mild         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E18 | ResNet18          |      100% | Strong       |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E19 | EfficientNet-B0   |       20% | None         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E20 | EfficientNet-B0   |       20% | Mild         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E21 | EfficientNet-B0   |       20% | Strong       |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E22 | EfficientNet-B0   |       50% | None         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E23 | EfficientNet-B0   |       50% | Mild         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E24 | EfficientNet-B0   |       50% | Strong       |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E25 | EfficientNet-B0   |      100% | None         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E26 | EfficientNet-B0   |      100% | Mild         |      TBD |       TBD |    TBD | TBD |                TBD |       |
| E27 | EfficientNet-B0   |      100% | Strong       |      TBD |       TBD |    TBD | TBD |                TBD |       |

## Notes

* The same train/validation/test split should be used consistently across experiments.
* Limited-data subsets should be generated consistently for fair comparison.
* Exact augmentation definitions and training hyperparameters should be documented once the training pipeline is finalized.
* Grad-CAM stability will be evaluated on a fixed class-balanced subset of test images.
