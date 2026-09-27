"""Grad-CAM heatmaps (Selvaraju et al. 2017, equations 1 and 2).

For each image, Grad-CAM asks which parts of the last convolutional layer's output pushed the
score of one class up. A forward hook captures that layer's activations A^k (k = channel, on a
7x7 grid for all three architectures); the gradient of the class score with respect to them is
taken with torch.autograd.grad. Equation 1 averages each channel's gradient over the grid into one
weight alpha_k; equation 2 is ReLU(sum_k alpha_k A^k), a 7x7 map of where the evidence for the
class sits. upsample() brings a map to image size. See report/papers/summaries.md for the
walkthrough of both equations, and NOTES.md for the decisions behind the choices below.

    from src.gradcam import gradcam, upsample
    maps, empty = gradcam(model, "resnet18", images, targets)    # (N, 7, 7) and (N,)
    heatmaps = upsample(maps, 224)                                 # (N, 224, 224)

Run it to check the implementation on all three architectures - against closed-form results
that do not use this module's code path - on the CPU, with no checkpoint:

    python -m src.gradcam

or to draw examples from a trained run's best checkpoint, with the randomisation sanity test.
Validation or training images only - development never looks at the test set:

    python -m src.gradcam --id T13 --output t13_gradcam.png [--per-class 4] [--split val]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from scipy.stats import spearmanr

from src.augment import IMAGE_SIZE, build_transform, denormalize
from src.config import get_config
from src.data import LABELS, ChestXrayDataset
from src.evaluate import classify
from src.figures import save_figure, titled
from src.models import ARCHITECTURES, build_model, gradcam_layer, set_fine_tuning, train_mode
from src.splits import load_split
from src.train import get_device, load_best, run_directory

# Class index -> name, the reverse of LABELS.
CLASS_NAMES = {index: name for name, index in LABELS.items()}
# The randomisation test's pass mark, stated before any real map was computed (NOTES.md): a
# faithful explanation of a trained model should share little rank order with one of a model
# whose weights are random.
RANDOM_MODEL_LIMIT = 0.5


def gradcam(model, name, images, targets):
    """Return (maps, empty): each image's Grad-CAM map for its target class, and which are all zero.

    images is a preprocessed (N, 3, 224, 224) batch on the model's device, targets the class
    index to explain for each image. maps is (N, 7, 7) float32 on the model's device, not
    normalised - src/stability.py compares raw maps. empty is an (N,) bool tensor, True where the
    ReLU left nothing, so every caller counts those the same way.

    The class score y^c is the pre-softmax logit, as in the paper. The model is put in eval mode
    here: images in a batch must not affect each other (BatchNorm on running statistics, dropout
    off) for one gradient over the batch to give each image its own - and under partial
    fine-tuning model.training alone does not show whether the last stage is in training mode.
    Results are reproducible to float tolerance, not bit for bit, on a GPU.
    """
    model.eval()
    captured = {}

    def keep(module, inputs, output):
        """Forward hook: store the Grad-CAM layer's output so its gradient can be taken."""
        # If every layer up to here is frozen, the output does not track gradients;
        # requires_grad_ switches that on before later layers use it, so the class score still
        # depends on it in the autograd graph.
        if not output.requires_grad:
            output.requires_grad_(True)
        captured["activations"] = output

    # register_forward_hook runs keep() every time the layer produces an output; the handle it
    # returns is how the hook is removed again.
    handle = gradcam_layer(model, name).register_forward_hook(keep)
    try:
        # enable_grad turns gradient tracking on even inside a caller's torch.no_grad().
        with torch.enable_grad():
            logits = model(images)
            activations = captured["activations"]
            # gather picks, from each row of logits, the column its target names: y^c per image.
            chosen = logits.gather(1, targets.to(logits.device).long().view(-1, 1))
            # Images are independent in eval mode, so the gradient of the summed scores holds
            # each image's own gradient. autograd.grad returns it without writing .grad into any
            # parameter, and stops at the activations instead of running back to the input.
            (gradients,) = torch.autograd.grad(chosen.sum(), activations)
    finally:
        handle.remove()
    # Equation 1: average each channel's gradient over the 7x7 grid - one weight per channel.
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    # Equation 2: weight each activation map, add them up over channels, keep the positive part.
    # .detach() drops the autograd history, which nothing downstream needs.
    maps = torch.relu((weights * activations).sum(dim=1)).detach().float()
    # .amax over the flattened grid is each map's largest value; 0 means the map is empty.
    empty = maps.flatten(1).amax(dim=1) == 0
    return maps, empty


def upsample(maps, size=IMAGE_SIZE):
    """Return (N, h, w) maps resized to (N, size, size) by bilinear interpolation, as float32."""
    # interpolate works on (batch, channels, height, width); unsqueeze(1) adds a channel axis
    # and squeeze(1) removes it again.
    return F.interpolate(maps.unsqueeze(1).float(), size=(size, size), mode="bilinear",
                         align_corners=False).squeeze(1)


def examples_figure(images, maps, empty, true, predicted, title, path, explained="predicted class"):
    """Draw each image above its Grad-CAM overlay, one column per image, and save it with its title.

    Each overlay is scaled to its own maximum, which is printed under it, so a weak map still
    shows where it points - and the number keeps the scale honest. An empty map is labelled so.
    explained names the class the maps explain, printed under the figure.
    """
    images, heat = images.cpu(), upsample(maps.cpu())
    figure = Figure(figsize=(2.1 * len(images), 4.6), layout="constrained")
    # squeeze=False keeps a 2D grid of axes even for a single image.
    grid = figure.subplots(2, len(images), squeeze=False)
    for column, image in enumerate(images):
        # The three channels are identical grayscale copies, so channel 0 is the X-ray.
        picture = denormalize(image)[0].numpy()
        top, bottom = grid[0, column], grid[1, column]
        top.imshow(picture, cmap="gray", vmin=0, vmax=1)
        top.set_title(f"true {CLASS_NAMES[int(true[column])]}\n"
                      f"predicted {CLASS_NAMES[int(predicted[column])]}", fontsize=7)
        bottom.imshow(picture, cmap="gray", vmin=0, vmax=1)
        # The raw 7x7 map's maximum - the number gradcam() returned and the CLI prints. Bilinear
        # upsampling never exceeds it, so scaling by it keeps the overlay within 0-1.
        peak = float(maps[column].max())
        # inferno runs dark to bright with even perceived steps, and reads in greyscale too.
        bottom.imshow(heat[column].numpy() / max(peak, 1e-12), cmap="inferno", alpha=0.45, vmin=0, vmax=1)
        bottom.set_xlabel("empty map" if bool(empty[column]) else f"max {peak:.3g}", fontsize=7)
        for axes in (top, bottom):
            axes.set_xticks([])
            axes.set_yticks([])
    figure.supxlabel(f"maps explain the {explained}", fontsize=7)
    titled(figure, title, size=8)
    return save_figure(figure, path, title)


def rank_agreement(maps, other, empty, other_empty):
    """Return (per-image Spearman correlations of two sets of maps on their raw cells, skipped count).

    Spearman compares the rank order of the 49 cells, so it ignores the maps' scale. An image is
    skipped when either map is empty - the correlation is undefined for a constant map.
    """
    values, skipped = [], 0
    for first, second, skip in zip(maps.cpu().flatten(1), other.cpu().flatten(1), (empty | other_empty).cpu()):
        # spearmanr(...)[0] is the correlation coefficient; [1] would be its p-value.
        value = np.nan if skip else spearmanr(first.numpy(), second.numpy())[0]
        if np.isnan(value):
            skipped += 1
        else:
            values.append(float(value))
    return values, skipped


def class_balanced_sample(labels, per_class, seed):
    """Return positions of per_class images of each class, drawn with a seeded generator."""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    chosen = []
    for index in sorted(CLASS_NAMES):
        # flatnonzero gives the positions where the condition holds.
        positions = np.flatnonzero(labels == index)
        chosen.extend(rng.choice(positions, size=per_class, replace=False).tolist())
    return chosen


def draw_examples(args):
    """Load a run's best checkpoint, draw its examples figure and print the sanity numbers."""
    config = get_config(args.id, args.seed)
    # get_device also sets cuDNN's deterministic flags, as for training and evaluation.
    device, device_name = get_device()
    model = load_best(config, run_directory(config, args.output_root), device).eval()

    # Evaluation preprocessing only, whatever regime the run trained with.
    dataset = ChestXrayDataset(load_split(args.split), build_transform(config.augmentation, train=False))
    positions = class_balanced_sample(dataset.labels, args.per_class, config.seed)
    images = torch.stack([dataset[position][0] for position in positions]).to(device)
    true = torch.tensor([dataset.labels[position] for position in positions])
    with torch.no_grad():
        predicted = torch.from_numpy(classify(model(images).cpu()))

    maps, empty = gradcam(model, config.model, images, predicted)
    title = f"{config.run_id} seed {config.seed} - {args.split} - Grad-CAM examples"
    examples_figure(images, maps, empty, true, predicted, title, args.output)
    print(f"{config.run_id} ({config.model}) on {device_name}: {len(positions)} {args.split} images, "
          f"figure {args.output}")
    for position, label, guess, peak, blank in zip(positions, true, predicted, maps.flatten(1).amax(1), empty):
        print(f"  {dataset.paths[position].name:32s} true {CLASS_NAMES[int(label)]:9s} "
              f"predicted {CLASS_NAMES[int(guess)]:9s} map max {float(peak):.4g}{'  EMPTY' if blank else ''}")

    # Model-randomisation test (Adebayo et al. 2018): if the maps mostly reflect the image and not
    # what the model learned, a model with random weights gives similar maps. The controls explain
    # the trained model's predicted classes, so only the weights differ.
    controls = {"random weights": False, "ImageNet weights, untrained head": True}
    for label, pretrained in controls.items():
        # Seeded, so the randomly initialised weights are the same on every run.
        torch.manual_seed(config.seed)
        control = build_model(config.model, pretrained=pretrained).to(device)
        other, other_empty = gradcam(control, config.model, images, predicted)
        values, skipped = rank_agreement(maps, other, empty, other_empty)
        summary = (f"median {np.median(values):+.3f}, range {min(values):+.3f} to {max(values):+.3f}"
                   if values else "no comparable pairs")
        verdict = ""
        if not pretrained and values:
            verdict = ("  -> pass" if np.median(values) < RANDOM_MODEL_LIMIT
                       else f"  -> INVESTIGATE (median not below {RANDOM_MODEL_LIMIT})")
        print(f"  vs {label}: Spearman on the 7x7 cells, {summary}; {len(values)} compared, "
              f"{skipped} skipped (empty map){verdict}")


def closed_form_maps(model, name, images, targets):
    """Return the Grad-CAM maps derived by hand from the head's weights - no hooks, no autograd.

    The activations come from calling the layers directly. ResNet18 and EfficientNet-B0 end in
    global average pooling and one linear layer (plus dropout, off in eval), so dy^c/dA_ij =
    w^c / 49 at every cell and Grad-CAM reduces to CAM: ReLU(sum_k w^c_k A^k) / 49 - the paper's
    "Grad-CAM generalises CAM" result. MobileNetV3-Small's head is Linear -> Hardswish ->
    Dropout -> Linear, whose gradient is written out with Hardswish's derivative.
    """
    with torch.no_grad():
        if name == "resnet18":
            x = model.maxpool(model.relu(model.bn1(model.conv1(images))))
            activations = model.layer4(model.layer3(model.layer2(model.layer1(x))))
            gradient = model.fc.weight[targets]
        else:
            # For both other architectures the Grad-CAM layer is the last block of .features.
            activations = model.features(images)
            if name == "efficientnet_b0":
                gradient = model.classifier[1].weight[targets]
            else:
                first, last = model.classifier[0], model.classifier[3]
                z = activations.mean(dim=(2, 3)) @ first.weight.T + first.bias
                # Hardswish(z) = z * relu6(z + 3) / 6; its slope is 0 below -3, 1 above 3, and
                # (2z + 3) / 6 in between.
                slope = torch.where(z < -3, torch.zeros_like(z),
                                    torch.where(z > 3, torch.ones_like(z), (2 * z + 3) / 6))
                gradient = (last.weight[targets] * slope) @ first.weight
        cells = activations.shape[2] * activations.shape[3]
        # einsum "nk,nkhw->nhw": for each image, a weighted sum of its channel maps.
        return torch.relu(torch.einsum("nk,nkhw->nhw", gradient / cells, activations))


def check_architecture(name):
    """Self-check gradcam() on one architecture against the closed form and a few invariants."""
    torch.manual_seed(0)
    model = build_model(name).eval()
    images = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    targets = torch.tensor([0, 1, 0, 1])
    maps, empty = gradcam(model, name, images, targets)
    expected = closed_form_maps(model, name, images, targets)
    # Relative error against the largest value, so the check does not depend on the maps' scale.
    error = float((maps - expected).abs().max() / expected.abs().max().clamp_min(1e-12))
    print(f"{name}:")
    print(f"  maps {tuple(maps.shape)}, upsampled {tuple(upsample(maps).shape)}, non-negative "
          f"{bool((maps >= 0).all())}, empty {int(empty.sum())} of {len(maps)}")
    print(f"  equals the closed form: {error < 1e-4} (relative difference {error:.1e})")

    one_by_one = torch.cat([gradcam(model, name, images[i:i + 1], targets[i:i + 1])[0] for i in range(4)])
    layer = gradcam_layer(model, name)
    # _forward_hooks is the layer's private dict of registered forward hooks.
    print(f"  batched equals one at a time: {torch.allclose(maps, one_by_one, rtol=1e-4, atol=1e-7)}; "
          f"no hook left: {len(layer._forward_hooks) == 0}; "
          f"no parameter .grad: {all(p.grad is None for p in model.parameters())}")

    other_class, _ = gradcam(model, name, images, 1 - targets)
    print(f"  the two classes' maps differ: {not torch.allclose(maps, other_class)}")

    # A partially fine-tuned model left in its training mode: gradcam must switch it to eval.
    set_fine_tuning(model, name, "partial")
    train_mode(model, name, "partial")
    partial, _ = gradcam(model, name, images, targets)
    # With every parameter frozen, the gradient must still reach the Grad-CAM layer.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frozen, _ = gradcam(model, name, images, targets)
    print(f"  same maps from partial training mode: {torch.allclose(maps, partial, rtol=1e-4, atol=1e-7)}; "
          f"with every layer frozen: {torch.allclose(maps, frozen, rtol=1e-4, atol=1e-7)}")


def main():
    """Self-check every architecture, or draw one run's examples when --id is given."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", help="run whose best checkpoint to explain, e.g. T13")
    parser.add_argument("--seed", type=int, default=None, help="the run's seed, if not its default")
    # The test split is deliberately not a choice: examples during development never use it.
    parser.add_argument("--split", default="val", choices=["val", "train_20", "train_50", "train_100"],
                        help="images to explain (default val; test is not offered)")
    parser.add_argument("--per-class", type=int, default=4, help="images of each class (default 4)")
    parser.add_argument("--output", type=Path, help="where to write the figure (required with --id)")
    parser.add_argument("--output-root", type=Path, default=None, help="where the run's checkpoints live")
    args = parser.parse_args()

    if args.id is None:
        for name in ARCHITECTURES:
            check_architecture(name)
        return
    if args.output is None:
        parser.error("--output is required with --id")
    draw_examples(args)


if __name__ == "__main__":
    main()
