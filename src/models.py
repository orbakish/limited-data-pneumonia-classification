"""Factory for the three architectures, each with a binary classification head.

MobileNetV3-Small, ResNet18 and EfficientNet-B0 come from torchvision with ImageNet weights;
only the final classifier layer is replaced. One file rather than three, because they differ
solely in which backbone is loaded - three near-identical files would drift apart, and the
whole experiment depends on the three being trained identically.

Also records the last convolutional layer of each architecture, which is where src/gradcam.py
attaches its hooks.

Not implemented yet - workplan section 4.
"""
