"""Deterministic preprocessing for every split, and the three augmentation regimes for training.

Two jobs, deliberately in one module because they compose into a single transform pipeline and
the boundary between them is the thing most easily got wrong:

* Preprocessing - applied to train, validation and test alike: grayscale -> 3 channels, squash
  to 224x224, ImageNet normalisation. Deterministic, so one image always becomes one tensor.
* Augmentation - applied to the training split only: none / mild / strong, exactly as defined
  in experiments/augmentation_plan.md. Random, and never applied to validation or test.

Noise and blur are never used here. They are reserved as the out-of-family perturbation in
src/stability.py, and training on them would make that measurement circular.

Not implemented yet - workplan section 5.
"""
