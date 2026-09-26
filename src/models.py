"""Factory for the three architectures, each with a two-logit classification head.

MobileNetV3-Small, ResNet18 and EfficientNet-B0 come from torchvision with ImageNet weights. Only
the final Linear layer is replaced, by a new one with two outputs (NORMAL = 0, PNEUMONIA = 1).
One file rather than three, because the architectures differ only in their ARCHITECTURES entry -
three near-identical files would drift apart, and the experiment depends on training them
identically.

Each ARCHITECTURES entry records, as module paths inside the model:

* head - the newly initialised output layer, the only part trained at the head learning rate.
* gradcam_layer - the last convolutional stage, where src/gradcam.py attaches its hooks.
* partial - what stays trainable under partial fine-tuning: every block at the final 7x7
  resolution (the resolution Grad-CAM reads) plus the classifier. That leaves 75 / 79 / 79% of
  the convolutional layers' parameters trainable, so the three are comparable (NOTES.md #6).
  The self-check reports 87% for MobileNetV3-Small because its count also includes the
  pretrained 576->1024 Linear layer in its classifier.

Import it:

    from src.models import build_model, set_batchnorm_momentum, set_fine_tuning, split_parameters, train_mode
    model = build_model("resnet18")
    set_fine_tuning(model, "resnet18", "partial")
    head, backbone = split_parameters(model, "resnet18")
    train_mode(model, "resnet18", "partial")   # in place of model.train(), every epoch

or run it to check the structure of all three (builds them without downloading weights):

    python -m src.models
"""

import torch
from torch import nn
from torchvision import models

# Two logits trained with cross-entropy, matching the labels in src/data.py.
NUM_CLASSES = 2

FINE_TUNING = ("full", "partial")

# Weights are pinned to IMAGENET1K_V1 rather than "DEFAULT", which a later torchvision release
# could point at a different checkpoint. Paths are dotted: "features.12" is features[12].
ARCHITECTURES = {
    "mobilenet_v3_small": {
        "builder": models.mobilenet_v3_small,
        "weights": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "head": "classifier.3",
        "gradcam_layer": "features.12",
        "partial": ("features.9", "features.10", "features.11", "features.12", "classifier"),
    },
    "resnet18": {
        "builder": models.resnet18,
        "weights": models.ResNet18_Weights.IMAGENET1K_V1,
        "head": "fc",
        "gradcam_layer": "layer4",
        "partial": ("layer4", "fc"),
    },
    "efficientnet_b0": {
        "builder": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "head": "classifier.1",
        "gradcam_layer": "features.8",
        "partial": ("features.6", "features.7", "features.8", "classifier"),
    },
}


def architecture(name):
    """Return the ARCHITECTURES entry for name, raising ValueError for an unknown name."""
    if name not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {name!r}, expected one of {list(ARCHITECTURES)}")
    return ARCHITECTURES[name]


def build_model(name, pretrained=True):
    """Return the named architecture with its output layer replaced by a two-logit Linear layer.

    pretrained=True loads the ImageNet weights (downloaded once, then read from torch's cache);
    pretrained=False leaves every layer randomly initialised, which is enough for structural
    checks. The new head is randomly initialised either way, so seed torch before calling this.
    """
    entry = architecture(name)
    model = entry["builder"](weights=entry["weights"] if pretrained else None)

    # "classifier.3".rpartition(".") gives ("classifier", ".", "3"); "fc" gives ("", "", "fc").
    parent_path, _, child_name = entry["head"].rpartition(".")
    # get_submodule follows a dotted path through the model; the empty path is the model itself.
    parent = model.get_submodule(parent_path)
    old_head = getattr(parent, child_name)
    # Assigning a Module as an attribute registers it in place of the old one - this also works
    # for nn.Sequential, whose children are stored under the names "0", "1", ...
    setattr(parent, child_name, nn.Linear(old_head.in_features, NUM_CLASSES))
    return model


def gradcam_layer(model, name):
    """Return the module whose activations and gradients Grad-CAM uses for this architecture."""
    return model.get_submodule(architecture(name)["gradcam_layer"])


def set_fine_tuning(model, name, mode):
    """Freeze or unfreeze parameters for "full" or "partial" fine-tuning.

    full trains every parameter; partial trains only the modules listed under "partial" and
    freezes the rest. Call it before building the optimizer, so the optimizer only receives
    parameters that train.
    """
    if mode not in FINE_TUNING:
        raise ValueError(f"unknown fine-tuning mode {mode!r}, expected one of {FINE_TUNING}")
    # requires_grad_ sets requires_grad on every parameter of the module and its children.
    model.requires_grad_(mode == "full")
    if mode == "partial":
        for path in architecture(name)["partial"]:
            model.get_submodule(path).requires_grad_(True)


def split_parameters(model, name):
    """Return (head, backbone): the new output layer's parameters, and every other trainable one.

    The two lists become the optimizer's two parameter groups, one per learning rate. Frozen
    parameters are left out of both.
    """
    head = list(model.get_submodule(architecture(name)["head"]).parameters())
    # A parameter tensor is compared by identity, not value: id() is unique per live object.
    head_ids = {id(parameter) for parameter in head}
    backbone = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    return head, backbone


def train_mode(model, name, mode):
    """Switch the model to training mode, keeping the frozen part in evaluation mode under partial.

    Freezing parameters does not freeze BatchNorm: in training mode it keeps updating its running
    mean and variance from every batch, so the frozen layers would drift differently under each
    augmentation regime. Evaluation mode stops that, and also switches off dropout and
    EfficientNet's stochastic depth in the frozen layers. Under partial, model.training itself
    reads False - check the layers, not that flag.
    """
    # train(False) is the same as eval(); both recurse into every child module.
    model.train(mode == "full")
    if mode == "partial":
        for path in architecture(name)["partial"]:
            model.get_submodule(path).train()


def set_batchnorm_momentum(model, momentum):
    """Set the running-statistics momentum of every BatchNorm layer and return how many changed.

    Momentum only decides how fast the running mean and variance - the statistics evaluation
    mode normalises with - follow the training batches. In training mode every batch is
    normalised with its own statistics, so the weights train identically at any momentum and
    only what evaluation sees changes. torchvision builds MobileNetV3 with 0.01, the other two
    with 0.1 (NOTES.md #26).
    """
    # _BatchNorm is the base class shared by BatchNorm1d, 2d and 3d. The leading underscore marks
    # it as torch-internal, but it is the one class every BatchNorm variant inherits from.
    norms = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    for norm in norms:
        norm.momentum = momentum
    return len(norms)



def count(parameters):
    """Return the total number of values across an iterable of parameters."""
    # .numel() is a tensor's element count.
    return sum(parameter.numel() for parameter in parameters)


def check_architecture(name):
    """Build one architecture without weights, run the structural checks, and print the results."""
    model = build_model(name, pretrained=False)
    images = torch.zeros(2, 3, 224, 224)

    shapes = {}
    # A forward hook runs after the layer's own forward pass and receives its output - the same
    # mechanism src/gradcam.py will use to capture activations.
    hook = gradcam_layer(model, name).register_forward_hook(
        lambda module, inputs, output: shapes.update(gradcam=tuple(output.shape)))
    model.eval()
    # no_grad skips recording the operations for backpropagation, which a check does not need.
    with torch.no_grad():
        logits = model(images)
    hook.remove()
    print(f"  logits {tuple(logits.shape)}, two per image: {tuple(logits.shape) == (2, NUM_CLASSES)}")
    print(f"  Grad-CAM layer output {shapes['gradcam']}, 7x7: {shapes['gradcam'][-2:] == (7, 7)}")

    head_size = count(model.get_submodule(architecture(name)["head"]).parameters())
    backbone_size = count(model.parameters()) - head_size
    for mode in FINE_TUNING:
        set_fine_tuning(model, name, mode)
        head, backbone = split_parameters(model, name)
        train_mode(model, name, mode)
        norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        # A BatchNorm layer should update its statistics exactly when its own weights train.
        norms_ok = all(norm.training == norm.weight.requires_grad for norm in norms)
        frozen = sum(not norm.training for norm in norms)
        print(f"  {mode:7s} trainable backbone {count(backbone):>10,} of {backbone_size:,} "
              f"({count(backbone) / backbone_size:.0%}), head {count(head):,}, "
              f"BatchNorm in eval: {frozen} of {len(norms)}, consistent: {norms_ok}")
    changed = set_batchnorm_momentum(model, 0.1)
    # Reads every BatchNorm layer back, not the return value, so a layer the function missed shows.
    applied = all(norm.momentum == 0.1 for norm in model.modules()
                  if isinstance(norm, nn.modules.batchnorm._BatchNorm))
    print(f"  BatchNorm momentum set to 0.1 on {changed} layers, all applied: {applied}")


def main():
    """Run the structural checks on all three architectures."""
    for name in ARCHITECTURES:
        print(f"--- {name} ---")
        check_architecture(name)


if __name__ == "__main__":
    main()
