"""Deterministic preprocessing for every split, and the three augmentation regimes for training.

Two jobs, deliberately in one module because they compose into a single transform pipeline and
the boundary between them is the thing most easily got wrong:

* Preprocessing - applied to train, validation and test alike: grayscale -> 3 channels, squash
  to 224x224, ImageNet normalisation. Deterministic, so one image always becomes one tensor.
* Augmentation - applied to the training split only: none / mild / strong, exactly as defined
  in experiments/augmentation_plan.md. Random, and never applied to validation or test.

Noise and blur are never used here. They are reserved as the out-of-family perturbation in
src/stability.py, and training on them would make that measurement circular.

Import it:

    from src.augment import build_transform
    train_tf = build_transform("mild", train=True)
    eval_tf = build_transform("mild", train=False)   # preprocessing only

or run it to self-check, write the quality-check sheets to data/processed/augment_preview/
(gitignored), and write the report figure to report/figures/augmentation_examples.png (committed):

    python -m src.augment
"""

from pathlib import Path

import torch
from matplotlib.figure import Figure
from PIL import Image
from torchvision.transforms import InterpolationMode, v2

from src.data import ChestXrayDataset
from src.paths import CLASSES, SPLITS, collect_images, data_root
from src.splits import SEED, load_split

# The pretrained models were trained on 224x224 inputs.
IMAGE_SIZE = 224

# Per-channel mean and standard deviation of the ImageNet training set. The pretrained weights
# expect inputs normalised with exactly these numbers, whatever the images actually are.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The magnitudes, mirroring the summary table in experiments/augmentation_plan.md. That document
# is the agreed definition: change both together, or the code and the report describe different
# experiments. translate is the maximum shift as a fraction of width/height; scale below 1 zooms
# out and above 1 zooms in, and the zoom-in cap is 1.05 in both regimes because zooming in is a
# crop. brightness/contrast 0.10 means a random factor between 0.90 and 1.10.
REGIMES = {
    "none": None,
    "mild": {"degrees": 5, "translate": 0.02, "scale": (0.95, 1.05), "brightness": 0.10, "contrast": 0.10},
    "strong": {"degrees": 15, "translate": 0.05, "scale": (0.85, 1.05), "brightness": 0.25, "contrast": 0.25},
}

# Images the preview must include: already cropped so tightly that any shift or zoom-in removes
# lung (NOTES.md #16). Filename stems, looked up across every split.
TIGHT_CROPS = ("person407_virus_811", "person325_virus_661", "IM-0022-0001")

REPO_ROOT = Path(__file__).resolve().parents[1]
# Quality-check sheets: working material showing many dataset images, so gitignored.
PREVIEW_DIR = REPO_ROOT / "data" / "processed" / "augment_preview"
# The report figure is committed, so the report always has a stable copy (NOTES.md, 2026-09-14).
REPORT_FIGURE = REPO_ROOT / "report" / "figures" / "augmentation_examples.png"


def resize_steps():
    """Return the deterministic steps that run before any augmentation, as a list of transforms."""
    return [
        # The Dataset already hands over single-channel "L" images; this replicates that one
        # channel three times, because the pretrained models expect RGB-shaped input.
        v2.Grayscale(num_output_channels=3),
        # A (height, width) pair squashes to exactly 224x224 instead of preserving aspect ratio -
        # see the resize decision in NOTES.md. antialias=True smooths before shrinking, which
        # avoids aliasing when a 1600-pixel image becomes 224.
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    ]


def augmentation_steps(regime):
    """Return a regime's random steps as [RandomAffine, ColorJitter], or [] for "none"."""
    params = REGIMES[regime]
    if params is None:
        return []
    return [
        # Rotation, shift and zoom in one affine transform, so the image is resampled once rather
        # than three times. It runs after the resize: rotating 224 pixels is about 30x cheaper
        # than rotating the original, and looks the same at the size the model sees.
        v2.RandomAffine(
            degrees=params["degrees"],
            translate=(params["translate"], params["translate"]),
            scale=params["scale"],
            # RandomAffine defaults to NEAREST, which leaves stair-stepped edges on rotated
            # anatomy - an artefact no real X-ray contains.
            interpolation=InterpolationMode.BILINEAR,
            # Pixels uncovered by rotating, shifting or zooming out become black - the same rule
            # for both classes, at a random place in every draw.
            fill=0,
        ),
        # Brightness scales every pixel; contrast stretches or compresses values around the image
        # mean. Applied identically to all three channels, so they stay equal.
        v2.ColorJitter(brightness=params["brightness"], contrast=params["contrast"]),
    ]


def tensor_steps():
    """Return the deterministic steps that run after augmentation, as a list of transforms."""
    return [
        # ToImage turns the PIL image into a uint8 tensor shaped (channels, height, width);
        # ToDtype with scale=True converts it to float and maps 0-255 onto 0.0-1.0.
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def build_transform(regime="none", train=False):
    """Return the transform pipeline for one augmentation regime, as a v2.Compose.

    With train=False, or regime "none", the pipeline is preprocessing only - which is how the
    rule "augmentation on the training split only" is enforced: validation and test loaders are
    simply built with train=False. Raises ValueError for an unknown regime name.
    """
    # Rejecting unknown names matters: a typo such as "strng" silently falling back to no
    # augmentation would produce a plausible-looking but mislabelled experiment.
    if regime not in REGIMES:
        raise ValueError(f"unknown augmentation regime {regime!r}; expected one of {sorted(REGIMES)}")
    random_steps = augmentation_steps(regime) if train else []
    # Compose runs the steps in list order, each on the output of the previous one. + on lists
    # concatenates them, so this is preprocessing, then augmentation if any, then tensor steps.
    return v2.Compose(resize_steps() + random_steps + tensor_steps())


def augment_once(image, regime):
    """Apply one random draw of a regime, and report the values that draw used.

    image must already be grayscale and resized (the output of resize_steps), since that is where
    augmentation sits in the training pipeline. Returns (tensor, caption): the tensor is exactly
    what build_transform(regime, train=True) produces from the same random state - main() checks
    this - and caption lists the rotation, shift, zoom, brightness and contrast that were drawn.
    """
    affine, jitter = augmentation_steps(regime)
    # A v2 transform normally does two things in one call: make_params draws this call's random
    # values, and transform applies them. Calling the two separately, in the same order the
    # pipeline would, gives an identical result while letting us read the values in between.
    affine_params = affine.make_params([image])
    augmented = affine.transform(image, affine_params)
    jitter_params = jitter.make_params([augmented])
    augmented = jitter.transform(augmented, jitter_params)
    tensor = v2.Compose(tensor_steps())(augmented)

    # translate comes back in pixels; dividing by the image size gives the fraction the regimes
    # are specified in. The format :+.1% writes 0.018 as "+1.8%", always with a sign.
    dx, dy = affine_params["translate"]
    caption = (
        f"rot {affine_params['angle']:+.1f}°   zoom {affine_params['scale']:.2f}\n"
        f"shift x {dx / IMAGE_SIZE:+.1%}  y {dy / IMAGE_SIZE:+.1%}\n"
        f"brightness {jitter_params['brightness_factor']:.2f}   contrast {jitter_params['contrast_factor']:.2f}"
    )
    return tensor, caption


def describe_regime(regime):
    """Return a one-line summary of a regime's ranges, for figure titles."""
    params = REGIMES[regime]
    if params is None:
        return f"{regime}: preprocessing only"
    low, high = params["scale"]
    return (
        f"{regime}: rotation ±{params['degrees']}°, shift up to {params['translate']:.0%}, "
        f"zoom {low:.2f}-{high:.2f}, brightness ±{params['brightness']:.0%}, contrast ±{params['contrast']:.0%}"
    )


def denormalize(tensor):
    """Undo the ImageNet normalisation, returning a (3, H, W) tensor clamped to 0-1 for display."""
    # .view(3, 1, 1) reshapes the three numbers so they broadcast across every pixel of each
    # channel - one value per channel, repeated over height and width.
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def find_preview_images(root):
    """Return [(label, path)] for the images the preview must show.

    The tight crops from NOTES.md #16, the widest and the shortest PNEUMONIA images (the extremes
    the survey flagged), and one typical image of each class from the validation split.
    """
    images = collect_images(root)
    everything = [path for split in SPLITS for cls in CLASSES for path in images[split][cls]]
    by_stem = {path.stem: path for path in everything}
    rows = [(f"tight crop: {stem}", by_stem[stem]) for stem in TIGHT_CROPS]

    pneumonia = [path for split in SPLITS for path in images[split]["PNEUMONIA"]]
    sizes = {}
    for path in pneumonia:
        # Image.open reads only the header until pixels are requested, so collecting the size of
        # ~4,000 images takes seconds rather than decoding all of them.
        with Image.open(path) as image:
            sizes[path] = image.size  # (width, height)
    # The most extreme images are the tight crops already listed above, so they are skipped:
    # the next-most-extreme ones show the preview something it would not otherwise see.
    already_shown = {path for _, path in rows}
    candidates = [path for path in pneumonia if path not in already_shown]
    # max/min with key= compare items by the value the key function returns, not by the items.
    widest = max(candidates, key=lambda path: sizes[path][0] / sizes[path][1])
    shortest = min((path for path in candidates if path != widest), key=lambda path: sizes[path][1])
    rows += [(f"widest: {widest.stem}", widest), (f"shortest: {shortest.stem}", shortest)]

    # val.txt is sorted, so its first entry is NORMAL and its last PNEUMONIA.
    validation = load_split("val", root)
    rows += [("typical NORMAL", validation[0]), ("typical PNEUMONIA", validation[-1])]
    return rows


def draw_figure(rows, columns, title, output_path, dpi=100, panel_size=2.0):
    """Render a grid of images to output_path: one row per image, one column per transform.

    rows is [(label, path)]. columns is [(header, regime)], where regime None shows the image
    preprocessed only. Every augmented panel is captioned with the values it was drawn with.
    """
    # Loading through ChestXrayDataset, not a separate Image.open, means the figure shows exactly
    # the grayscale images training will see.
    dataset = ChestXrayDataset([path for _, path in rows])
    resize = v2.Compose(resize_steps())
    to_tensor = v2.Compose(tensor_steps())

    # layout="constrained" lets matplotlib space the panels, captions and multi-line title so
    # they don't overlap - more reliable than tight_layout when there is a suptitle.
    figure = Figure(figsize=(panel_size * len(columns), panel_size * 1.3 * len(rows)), layout="constrained")
    for row, (label, _) in enumerate(rows):
        image = resize(dataset[row][0])
        for column, (header, regime) in enumerate(columns):
            if regime is None:
                tensor, caption = to_tensor(image), "(no augmentation)"
            else:
                tensor, caption = augment_once(image, regime)
            # add_subplot(rows, columns, n) places a panel in a grid numbered left to right,
            # top to bottom, starting at 1.
            axis = figure.add_subplot(len(rows), len(columns), row * len(columns) + column + 1)
            # All three channels are identical, so channel 0 alone shows the image.
            axis.imshow(denormalize(tensor)[0].numpy(), cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel(caption, fontsize=6)
            if column == 0:
                axis.set_ylabel(label, fontsize=7)
            if row == 0:
                axis.set_title(header, fontsize=8)
    figure.suptitle(title, fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)


def main():
    """Self-check the transforms, then render the quality-check sheets and the report figure."""
    rows = find_preview_images(data_root())
    image, _ = ChestXrayDataset([rows[-1][1]])[0]

    # 1. Preprocessing must be deterministic: the same image twice gives the same tensor.
    evaluate = build_transform("strong", train=False)
    first, second = evaluate(image), evaluate(image)
    print(f"preprocessing deterministic: {torch.equal(first, second)}")
    print(f"output: shape {tuple(first.shape)}, dtype {first.dtype}")

    # 2. The three channels must carry the same image. Compared after undoing the normalisation,
    # because Normalize uses a different mean and std per channel, so the normalised values differ
    # even when the underlying image is identical.
    shown = denormalize(first)
    # atol=1e-6 because normalising and then undoing it in float32 leaves rounding error around 1e-7;
    # allclose's default atol of 1e-8 is stricter than float32 can represent near zero.
    same = torch.allclose(shown[0], shown[1], atol=1e-6) and torch.allclose(shown[1], shown[2], atol=1e-6)
    print(f"channels identical: {same}")

    # 3. Augmentation must actually vary: two draws from the training pipeline should differ.
    torch.manual_seed(SEED)
    train = build_transform("strong", train=True)
    print(f"augmentation varies between draws: {not torch.equal(train(image), train(image))}")

    # 4. The captions must describe what training really does: from the same random state, one
    # captioned draw has to equal one pass through the training pipeline, bit for bit.
    resized = v2.Compose(resize_steps())(image)
    torch.manual_seed(SEED)
    via_pipeline = build_transform("strong", train=True)(image)
    torch.manual_seed(SEED)
    via_captions, _ = augment_once(resized, "strong")
    print(f"captioned draws match the training pipeline: {torch.equal(via_pipeline, via_captions)}")

    # Quality-check sheets: every preview image, five draws per regime.
    for regime in ("mild", "strong"):
        # Reseeded per figure, so rerunning draws exactly the same augmentations.
        torch.manual_seed(SEED)
        columns = [("preprocessed", None)] + [(f"draw {number}", regime) for number in range(1, 6)]
        output_path = PREVIEW_DIR / f"preview_{regime}.png"
        draw_figure(rows, columns, f"QA preview - {describe_regime(regime)} - seed {SEED}", output_path)
        print(f"saved {output_path}")

    # Report figure: the two typical images only (the last two rows), both regimes side by side,
    # larger panels and 200 dpi for print.
    torch.manual_seed(SEED)
    columns = [
        ("preprocessed", None),
        ("mild, draw 1", "mild"),
        ("mild, draw 2", "mild"),
        ("strong, draw 1", "strong"),
        ("strong, draw 2", "strong"),
    ]
    title = (
        "Augmentation regimes, applied to the training split only\n"
        + describe_regime("mild") + "\n" + describe_regime("strong")
    )
    draw_figure(rows[-2:], columns, title, REPORT_FIGURE, dpi=200, panel_size=2.6)
    print(f"saved {REPORT_FIGURE}")


if __name__ == "__main__":
    main()
