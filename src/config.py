"""Experiment configuration: the (model, data size, augmentation) triple and its E01-E27 ID.

Resolves in both directions against the table in experiments/experiment_log.md. IDs run
model-major, then data size, then augmentation regime - so E01 is MobileNetV3-Small / 20% /
none, and E27 is EfficientNet-B0 / 100% / strong. Also carries the random seed, so every run
records exactly what produced it.

Not implemented yet - workplan section 3.
"""
