# Experiment Log

This file tracks the main training configurations and their results.

| ID  | Model             | Data Size | Augmentation | Accuracy | Precision | Recall | Specificity | F1 | Balanced Accuracy | ROC-AUC | Grad-CAM Stability | Notes |
| --- | ----------------- | --------: | ------------ | -------: | --------: | -----: | ----------: | -: | ----------------: | ------: | -----------------: | ----- |
| E01 | MobileNetV3-Small |       20% | None         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E02 | MobileNetV3-Small |       20% | Mild         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E03 | MobileNetV3-Small |       20% | Strong       |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E04 | MobileNetV3-Small |       50% | None         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E05 | MobileNetV3-Small |       50% | Mild         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E06 | MobileNetV3-Small |       50% | Strong       |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E07 | MobileNetV3-Small |      100% | None         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E08 | MobileNetV3-Small |      100% | Mild         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E09 | MobileNetV3-Small |      100% | Strong       |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E10 | ResNet18          |       20% | None         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E11 | ResNet18          |       20% | Mild         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E12 | ResNet18          |       20% | Strong       |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E13 | ResNet18          |       50% | None         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E14 | ResNet18          |       50% | Mild         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E15 | ResNet18          |       50% | Strong       |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E16 | ResNet18          |      100% | None         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E17 | ResNet18          |      100% | Mild         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E18 | ResNet18          |      100% | Strong       |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E19 | EfficientNet-B0   |       20% | None         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E20 | EfficientNet-B0   |       20% | Mild         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E21 | EfficientNet-B0   |       20% | Strong       |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E22 | EfficientNet-B0   |       50% | None         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E23 | EfficientNet-B0   |       50% | Mild         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E24 | EfficientNet-B0   |       50% | Strong       |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E25 | EfficientNet-B0   |      100% | None         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E26 | EfficientNet-B0   |      100% | Mild         |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |
| E27 | EfficientNet-B0   |      100% | Strong       |      TBD |       TBD |    TBD |          TBD| TBD|                TBD|      TBD|                 TBD|       |

## Notes

* The same train/validation/test split is used consistently across all experiments.
* Limited-data subsets are generated consistently for fair comparison.
* All E01–E27 experiments use partial fine-tuning and class-weighted cross-entropy.
* Training uses the frozen recipe selected during the validation-only tuning phase: up to 30 epochs with early stopping patience 5.
* The main threshold-dependent metrics in this table use the threshold selected on validation to maximize balanced accuracy. The corresponding metrics at threshold 0.5 are recorded in the Notes column.
* Checkpoint selection remains based on validation balanced accuracy at threshold 0.5.
* Grad-CAM stability is evaluated on a fixed class-balanced subset of test images.
* Grad-CAM stability is reported as Pearson correlation in the format `in-family / out-of-family (gap)`.
