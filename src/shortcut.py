"""Shortcut check (NOTES.md #15): do the models follow image properties that owe nothing to pathology?

Planned and reviewed before any model output was related to an image property; the readings were
stated in the plan and are recorded in NOTES.md #15. Two dataset facts shaped it: the JPEG encoding
differs between the classes in training (healthy images saved at about quality 95, pneumonia at 75)
but not on test, where every healthy image is at 75; and at 224x224 the healthy images, not the
pneumonia ones, are the grainier. Four parts:

* covariates (local, CPU): for every image of train_100, val and test, properties unrelated to
  pathology - the encoding, the original size and shape, and at 224x224 the mean intensity,
  contrast, dark fraction and fine grain; how well each alone separates the classes, and whether
  the models' margins follow any of them within a class, given all the others (partial Spearman).
* regions (local): where each saved true-class Grad-CAM map puts its mass - on the image's
  background cells, in the corners and bottom row - and where it peaks, per run and class, with
  paired comparisons between regimes and data sizes.
* intervention (Colab, GPU): every checkpoint scores validation and test again in four versions -
  unperturbed (checked against the collected scores first), re-encoded as JPEG quality 75, blurred,
  and sharpened by the same amount - and gives margin Grad-CAM maps of the stability subset. Its
  readings, where the margin maps sit and each version's pixel change are computed locally, with the rest.
* audit (manual): numbered overlays for a person to label where each map's hot spot sits.

    python -m src.shortcut                   # covariates, regions, intervention readings -> results/shortcut/
    python -m src.shortcut --audit-sheets    # the audit's numbered sheets and label template
    python -m src.shortcut --intervene --seed 42 --workers 2   # Colab; add --id E01 for one run
    python -m src.shortcut --check           # self-check on synthetic data
"""

import argparse
import csv
import io
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from matplotlib.figure import Figure
from PIL import Image
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from src.analysis import (SEEDS, direction_lines, experiment_ids, image_names, interval, load_runs, paired_mean,
                          pairs, subset_name, subset_weights, views, write_csv)
from src.augment import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_transform, resize_steps, tensor_steps
from src.config import (AUGMENTATIONS, EXPERIMENTS, LOG_LABELS, MODELS, RECIPE, TRAIN_SPLITS, get_config,
                        output_stem)
from src.data import LABELS, ChestXrayDataset
from src.evaluate import classify, margins, results_directory
from src.figures import REGIME_COLOURS, REGIME_MARKERS, save_figure, titled
from src.gradcam import MARGIN, gradcam, upsample
from src.paths import code_version
from src.splits import SEED, load_split
from src.train import get_device, load_best, run_directory

# Or's EDA inventory: every image's original width, height and file size (data/eda/README.md).
INVENTORY = Path(__file__).resolve().parents[1] / "data" / "eda" / "image_inventory.csv"
SPLIT_FILES = ("train_100", "val", "test")
# A pixel darker than this, on the 0-1 scale, counts as background: the black surround of the body.
DARK = 0.1
# Grain is what a slight Gaussian blur removes; a kernel of 7 covers +-3 sigma.
GRAIN_SIGMA, GRAIN_KERNEL = 1.0, 7
# The grain is measured on the central 160x160 pixels, which drops exactly the outer ring of the 7x7
# map cells (32 pixels each) - the background and framing the regions part measures on its own.
CROP = 160
CELLS = 7
# The margins follow a covariate only if its median partial |Spearman| reaches this (NOTES.md #15).
MIN_RHO = 0.20
# (column, how summary.md names it). The partial correlations use these eight; grain's horizontal
# and vertical parts are a sensitivity check only, since grain already contains them.
COVARIATES = (
    ("quant", "JPEG quantisation-table mean"),
    ("bytes_per_pixel", "file bytes per pixel"),
    ("megapixels", "original megapixels"),
    ("aspect", "original aspect ratio (w/h)"),
    ("mean", "mean intensity at 224"),
    ("contrast", "contrast (std) at 224"),
    ("dark", "dark fraction at 224"),
    ("grain", "grain at 224"),
)
DIRECTIONS = (("grain_h", "grain at 224, horizontal"), ("grain_v", "grain at 224, vertical"))
NUMERIC = tuple(name for name, _ in COVARIATES + DIRECTIONS)
# The cells where a side marker or the region below the diaphragm fall: the top corners and the bottom
# row. The rest of the top row is left out: it holds the lung apices.
EDGE = np.zeros((CELLS, CELLS), dtype=bool)
EDGE[0, 0] = EDGE[0, CELLS - 1] = True
EDGE[CELLS - 1] = True
REGION_MEASURES = (
    ("background_share", "", "share of the map on background cells"),
    ("edge_share", "", "share on the top corners and bottom row"),
    ("peak_background", "", "peak on a background cell"),
)
# The audit: 5 images of each class, the same for the six runs; 12 overlays to a sheet.
AUDIT_PER_CLASS, AUDIT_PER_SHEET = 5, 12
AUDIT_LABELS = ("lung field", "mediastinum-heart", "image edge-background", "marker-text", "device-wires",
                "below the diaphragm", "diffuse-other")
# The intervention's four versions of every image. The blur is the grain measure's blur, and the sharpen
# adds back what that blur removes, so the two move the pixels by the same amount in opposite directions -
# before the clip to 0-1, which trims the sharpen where the image is black or white.
PASSES = ("unperturbed", "reencoded", "blur", "sharpen")
VERSIONS = PASSES[1:]
# The healthy test images' JPEG quality: PIL's luminance table at 75 is theirs, entry for entry (self_check).
QUALITY = 75
# That table's mean, as covariates.csv records it for every healthy test image.
Q75_TABLE = 29.03125
# The gate's approximate tier: every logit within this of the collected one, and no prediction changed.
GATE_APPROX = 1e-5
# The intervention's readings, stated in the plan before any perturbed score was read (NOTES.md #15): the
# encoding is supported where the healthy validation margins rise by at least ENCODING_RISE logits and
# ENCODING_RATIO times the pneumonia images' change, not supported where they rise by less than ENCODING_WEAK -
# in at least VERDICT_RUNS of an architecture's 18 runs (9 configurations x 2 seeds).
ENCODING_RISE, ENCODING_RATIO, ENCODING_WEAK, VERDICT_RUNS = 1.0, 5.0, 0.5, 12
# No share of the validation-to-test specificity gap is shown for a gap below this - about 12 of 234 test images.
MIN_GAP = 0.05
# The audit's margin overlays are drawn only if fewer than a quarter of their 30 maps are empty: at most 7.
AUDIT_MAX_EMPTY = 7


def unnormalised_transform():
    """Return the evaluation preprocessing without its normalisation: a (3, 224, 224) tensor in 0-1.

    tensor_steps() ends with Normalize; its first two steps turn the resized image into floats in
    0-1. Normalising this output gives exactly what the models saw (checked in self_check).
    """
    return v2.Compose(resize_steps() + tensor_steps()[:2])


def image_features(image):
    """Return the model-free properties of one preprocessed image as a dict of floats.

    image is a (224, 224) tensor in 0-1. Grain is the standard deviation of what a slight blur
    removes, on the central crop; its horizontal and vertical parts blur along one axis only.
    """
    def residual(kernel):
        """Return the image minus its blur with the given (width, height) kernel."""
        # gaussian_blur takes (..., height, width) tensors and the kernel as (width, height); a
        # size of 1 leaves that axis untouched.
        return image - TF.gaussian_blur(image[None], kernel_size=kernel, sigma=[GRAIN_SIGMA, GRAIN_SIGMA])[0]

    start = (IMAGE_SIZE - CROP) // 2
    # A tuple of two slices indexes rows and columns at once: the central square.
    centre = (slice(start, start + CROP), slice(start, start + CROP))
    return {
        "mean": float(image.mean()), "contrast": float(image.std()), "dark": float((image < DARK).float().mean()),
        "grain": float(residual([GRAIN_KERNEL, GRAIN_KERNEL])[centre].std()),
        "grain_h": float(residual([GRAIN_KERNEL, 1])[centre].std()),
        "grain_v": float(residual([1, GRAIN_KERNEL])[centre].std()),
    }


def background_cells(image):
    """Return a (7, 7) bool array: the map cells more than half of whose pixels are dark."""
    size = IMAGE_SIZE // CELLS
    # reshape splits each axis into (cell, pixel within the cell); the mean over the two pixel axes
    # is each cell's share of dark pixels.
    dark = (image < DARK).float().reshape(CELLS, size, CELLS, size).mean(dim=(1, 3))
    return (dark > 0.5).numpy()


def quantisation_mean(path):
    """Return the mean of a JPEG file's luminance quantisation table: the lower, the higher the quality."""
    with Image.open(path) as source:
        # .quantization is PIL's dict of a JPEG's quantisation tables; table 0 is luminance. getattr
        # gives None for a file that is not a JPEG, so such a file yields NaN instead of an error.
        tables = getattr(source, "quantization", None)
    return float(np.mean(tables[0])) if tables else float("nan")


def read_rows(path, numeric):
    """Return a CSV's rows as dicts, with the named columns turned back into floats."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update({name: float(row[name]) for name in numeric})
    return rows


def covariates(root=None):
    """Return one row per image of train_100, val and test: its model-free properties.

    Every image is decoded and resized once (a few minutes); the rows are kept in
    results/shortcut/covariates.csv and read back afterwards. Delete that file to compute them again.
    """
    path = results_directory(root, "shortcut") / "covariates.csv"
    if path.exists():
        return read_rows(path, NUMERIC)
    with INVENTORY.open(encoding="utf-8", newline="") as handle:
        sizes = {row["path"]: row for row in csv.DictReader(handle)}
    rows = []
    for split in SPLIT_FILES:
        dataset = ChestXrayDataset(load_split(split), unnormalised_transform())
        for index, file in enumerate(dataset.paths):
            # "train/NORMAL/IM-0115-0001.jpeg": the form the split lists and the inventory use.
            key = "/".join(file.parts[-3:])
            width, height = int(sizes[key]["width"]), int(sizes[key]["height"])
            # The three channels are identical grayscale copies, so channel 0 is the X-ray.
            image = dataset[index][0][0]
            row = {"split": split, "class": file.parent.name, "image": f"{file.parent.name}/{file.name}",
                   "file": key, "quant": quantisation_mean(file),
                   "bytes_per_pixel": int(sizes[key]["file_size"]) / (width * height),
                   "megapixels": width * height / 1e6, "aspect": width / height}
            row.update(image_features(image))
            row["background"] = "".join("1" if cell else "0" for cell in background_cells(image).flatten())
            rows.append(row)
        print(f"  covariates of {split}: {len(dataset)} images", flush=True)
    write_csv(rows, path)
    return rows


def availability(rows):
    """Return one row per covariate: its ROC-AUC as a lone classifier, three ways.

    Between the classes, in the training pool and on test, higher values meaning PNEUMONIA: 0.5 is
    no separation and below 0.5 the covariate is higher for NORMAL. Between healthy images, higher
    values meaning a test image rather than a training or validation one - the shift of NOTES.md #33.
    """
    def auc(group, positive, name):
        """Return the ROC-AUC of one covariate for telling the positive rows from the rest."""
        return float(roc_auc_score([positive(row) for row in group], [row[name] for row in group]))

    training = [row for row in rows if row["split"] == "train_100"]
    test = [row for row in rows if row["split"] == "test"]
    healthy = [row for row in rows if row["class"] == "NORMAL"]
    return [{"covariate": name, "label": label,
             "classes_training": auc(training, lambda row: row["class"] == "PNEUMONIA", name),
             "classes_test": auc(test, lambda row: row["class"] == "PNEUMONIA", name),
             "healthy_shift": auc(healthy, lambda row: row["split"] == "test", name)}
            for name, label in COVARIATES + DIRECTIONS]


def spearman(first, second):
    """Return the Spearman correlation of two arrays: the Pearson correlation of their ranks."""
    # rankdata gives tied values their average rank, as Spearman's definition asks.
    return float(np.corrcoef(rankdata(first), rankdata(second))[0, 1])


def partial_spearman(target, columns, index):
    """Return the Spearman correlation of columns[index] with target, given all the other columns.

    Everything becomes ranks; the other columns' linear effect is removed from the target's ranks
    and from this column's ranks by least squares, and what is left of the two is correlated. NaN
    when this column has nothing left once the others are removed.
    """
    ranked = np.column_stack([rankdata(column) for column in columns])
    design = np.column_stack([np.ones(len(target))] + [ranked[:, other] for other in range(ranked.shape[1])
                                                        if other != index])

    def residual(values):
        """Return what the other columns cannot explain of values."""
        # lstsq finds the coefficients that make design @ coefficients closest to values.
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        return values - design @ coefficients

    one, two = residual(ranked[:, index]), residual(rankdata(target))
    if one.std() < 1e-9 or two.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(one, two)[0, 1])


def run_columns(run):
    """Return the columns that name a run by its configuration, as in results/analysis/runs.csv."""
    config = run["config"]
    return {"experiment": run["id"], "seed": run["seed"], "model": LOG_LABELS[config.model],
            "data_size": LOG_LABELS[config.split], "augmentation": LOG_LABELS[config.augmentation]}


def margin_correlations(runs, rows):
    """Return one row per run, split, class and covariate: its Spearman correlation with the margin.

    Within one class, alone and given the other covariates. A covariate that is the same for every
    image of a group - the encoding of the healthy test images, all quality 75 - says nothing within
    it and is left out of that group's partial correlations.
    """
    by_split = {split: {row["image"]: row for row in rows if row["split"] == split} for split in ("val", "test")}
    results = []
    for run in runs.values():
        for split, names, logits, labels in (("test", run["images"], run["logits"], run["labels"]),
                                             ("val", run["val_images"], run["val_logits"], run["val_labels"])):
            margin = margins(logits)
            for name, index in LABELS.items():
                keep = labels.numpy() == index
                chosen = [by_split[split][image] for image, kept in zip(names, keep) if kept]
                present = [column for column, _ in COVARIATES if len({row[column] for row in chosen}) > 1]
                columns = [np.array([row[column] for row in chosen]) for column in present]
                base = {**run_columns(run), "split": split, "class": name}
                for position, column in enumerate(present):
                    results.append({**base, "covariate": column, "spearman": spearman(columns[position], margin[keep]),
                                    "partial": partial_spearman(margin[keep], columns, position)})
                for column, _ in DIRECTIONS:
                    values = np.array([row[column] for row in chosen])
                    results.append({**base, "covariate": column, "spearman": spearman(values, margin[keep]),
                                    "partial": float("nan")})
    return results


def named_covariate(cells):
    """Return the covariate the margins follow in one split and class, or None - the plan's rule.

    A covariate is named only if its median partial |Spearman| reaches MIN_RHO in both seeds and
    is larger, in each seed, than every other covariate's. cells are margin_correlations rows of
    one split and class.
    """
    medians = {}
    for column, _ in COVARIATES:
        values = [[row["partial"] for row in cells if row["covariate"] == column and row["seed"] == seed
                   and not np.isnan(row["partial"])] for seed in SEEDS]
        if all(values):
            medians[column] = [abs(float(np.median(value))) for value in values]
    for column, pair in medians.items():
        others = [other for other in medians if other != column]
        if all(pair[i] >= MIN_RHO and all(pair[i] > medians[other][i] for other in others) for i in range(len(SEEDS))):
            return column
    return None


def shift_correlations(runs, rows):
    """Return, per architecture, regime and seed, each covariate's Spearman correlation with the healthy
    test images' margin increase from 20% to 100% of the data (descriptive, NOTES.md #33)."""
    ids = experiment_ids()
    test = {row["image"]: row for row in rows if row["split"] == "test"}
    results = []
    for model in MODELS:
        for regime in AUGMENTATIONS:
            for seed in SEEDS:
                small = runs[ids[model, TRAIN_SPLITS[0], regime], seed]
                large = runs[ids[model, TRAIN_SPLITS[-1], regime], seed]
                healthy = small["labels"].numpy() == LABELS["NORMAL"]
                change = (margins(large["logits"]) - margins(small["logits"]))[healthy]
                chosen = [test[image] for image, kept in zip(small["images"], healthy) if kept]
                for column, _ in COVARIATES:
                    values = np.array([row[column] for row in chosen])
                    rho = spearman(values, change) if len(set(values)) > 1 else float("nan")
                    results.append({"model": LOG_LABELS[model], "augmentation": LOG_LABELS[regime], "seed": seed,
                                    "covariate": column, "spearman": rho})
    return results


def map_regions(maps, background):
    """Return (background share, edge share, peak category) for each of a stack of maps.

    maps is (N, 7, 7) and background (N, 7, 7) bool. A share is the map's mass on those cells over
    its total mass; edge is EDGE. The peak is the cell with the most mass, named "background" if it is
    a background cell, else "edge" if it is in EDGE, else "other". An empty map has NaN shares and
    the category "empty".
    """
    flat, back = maps.reshape(len(maps), -1), background.reshape(len(maps), -1)
    mass = flat.sum(axis=1)
    empty = mass <= 0
    # Dividing by 1 where a map is empty avoids a division by zero; those shares become NaN anyway.
    safe = np.where(empty, 1.0, mass)
    background_share = np.where(empty, np.nan, (flat * back).sum(axis=1) / safe)
    edge_share = np.where(empty, np.nan, (flat * EDGE.flatten()).sum(axis=1) / safe)
    peaks = flat.argmax(axis=1)
    # back[np.arange(N), peaks] reads, for each map, whether its own peak cell is a background cell.
    category = np.where(back[np.arange(len(maps)), peaks], "background",
                        np.where(EDGE.flatten()[peaks], "edge", "other"))
    return background_share, edge_share, np.where(empty, "empty", category)


def region_arrays(runs, rows, root=None, source="true class"):
    """Return {(ID, seed): per-image region values} of every run's clean true-class maps, or its margin maps.

    The true-class maps are results/runs/<run>/stability_maps.npy[:, 0], the margin maps (source "margin")
    margin_maps.npy - both the 468 subset images, in the order of the run's stability records - and each
    image's background cells come from covariates.csv.
    """
    test = {row["image"]: row for row in rows if row["split"] == "test"}
    arrays = {}
    for key, run in runs.items():
        names = [subset_name(record) for record in run["records"]]
        # Each image's 49 background flags, stored as a string of 0s and 1s, back to a 7x7 grid.
        background = np.array([[cell == "1" for cell in test[name]["background"]] for name in names])
        folder = results_directory(root, "runs") / run_directory(run["config"]).name
        if source == "true class":
            # stability_maps.npy holds five maps per image, the clean one first.
            maps = np.load(folder / "stability_maps.npy")[:, 0]
        else:
            maps = np.load(folder / "margin_maps.npy")
        share, edge, category = map_regions(maps, background.reshape(-1, CELLS, CELLS))
        arrays[key] = {"background_share": share, "edge_share": edge,
                       "peak_background": np.where(category == "empty", np.nan, (category == "background") * 1.0),
                       "empty": category == "empty", "class": np.array([name.split("/")[0] for name in names])}
    return arrays


def region_rows(runs, arrays, weights):
    """Return one row per run and class: each region measure's mean with its interval, and the empty maps."""
    results = []
    for key, run in runs.items():
        for name in LABELS:
            in_class = arrays[key]["class"] == name
            kept = in_class & ~arrays[key]["empty"]
            row = {**run_columns(run), "class": name, "maps": int(kept.sum()),
                   "empty": int((in_class & arrays[key]["empty"]).sum())}
            for measure, _, _ in REGION_MEASURES:
                # paired_mean averages per-image values over the kept images, on every resample too;
                # nan_to_num turns the empty maps' NaN into 0, which kept then leaves out.
                mean, draws, _ = paired_mean(np.nan_to_num(arrays[key][measure]), kept, weights)
                low, high = interval(draws)
                row.update({measure: mean, f"{measure}_low": low, f"{measure}_high": high})
            results.append(row)
    return results


def region_comparisons(arrays, weights):
    """Return comparison rows for every regime and data-size pair (analysis.pairs), per class and measure.

    Each is second run minus first, on the images of the class whose maps are non-empty in both, per
    seed and as the two seeds' mean, with the flags of NOTES.md #32 - read as results/analysis is.
    """
    results = []
    for kind, contrast, fixed, first, second in pairs():
        for name in LABELS:
            for measure, variant, _ in REGION_MEASURES:
                per_view = {}
                for seed in SEEDS:
                    one, two = arrays[first, seed], arrays[second, seed]
                    kept = (one["class"] == name) & ~one["empty"] & ~two["empty"]
                    per_view[f"seed {seed}"] = paired_mean(np.nan_to_num(two[measure]) - np.nan_to_num(one[measure]),
                                                           kept, weights)
                for view in views(per_view):
                    results.append({"kind": kind, "contrast": contrast, **fixed, "class": name, "first": first,
                                    "second": second, "metric": measure, "variant": variant, **view})
    return results


def preprocessing_check(count=16):
    """Return the largest difference between the evaluation transform and this module's, normalised.

    Checked on the first test images: the covariates must describe exactly what the models saw.
    """
    normalise = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    paths = load_split("test")[:count]
    own = ChestXrayDataset(paths, unnormalised_transform())
    theirs = ChestXrayDataset(paths, build_transform("none", train=False))
    return max(float((normalise(own[index][0]) - theirs[index][0]).abs().max()) for index in range(count))


def median_line(values):
    """Return 'median (low to high)' of a list, or '-' when it is empty."""
    values = [value for value in values if not np.isnan(value)]
    if not values:
        return "-"
    return f"{np.median(values):+.3f} ({min(values):+.2f} to {max(values):+.2f})"


def summary(rows, table, correlations, shifts, per_run, comparisons, difference):
    """Return results/shortcut/summary.md: the checks, the covariates, the margins and the regions."""
    counts = {split: sum(row["split"] == split for row in rows) for split in SPLIT_FILES}
    lines = [
        "# Shortcut check (NOTES.md #15) - covariates and regions",
        "",
        "Written by `python -m src.shortcut` from the dataset and `results/runs/`; change the code, not this file.",
        "The readings were stated in the plan before any model output was related to an image property (NOTES.md",
        "#15). The intervention (Colab) follows the regions, from the section \"The intervention (part 3)\" on; the",
        "audit's labels come later.",
        "",
        "## Checks",
        "",
        f"- Covariates of {counts['train_100']} training, {counts['val']} validation and {counts['test']} test images.",
        f"- The module's preprocessing, normalised, against the evaluation transform: largest difference "
        f"{difference:.1e} on 16 test images.",
        f"- Empty true-class maps, all runs: {sum(row['empty'] for row in per_run)} of "
        f"{sum(row['empty'] + row['maps'] for row in per_run)}.",
        "",
        "## What the data offers: each covariate alone as a classifier (ROC-AUC)",
        "",
        "Between the classes, higher values meaning PNEUMONIA (0.5 is no separation, below 0.5 the covariate is",
        "higher for NORMAL); between healthy images, higher values meaning a test image rather than a training or",
        "validation one (the shift of NOTES.md #33).",
        "",
        "| Covariate | Classes, training pool | Classes, test | Healthy, test against training and validation |",
        "|---|---|---|---|",
    ]
    lines += [f"| {row['label']} | {row['classes_training']:.3f} | {row['classes_test']:.3f} | {row['healthy_shift']:.3f} |"
              for row in table]
    lines += ["", "## Do the margins follow them? Within one class, over the runs of each seed", "",
              "Spearman correlation of each covariate with the logit margin (PNEUMONIA minus NORMAL), alone and given",
              "the other seven (partial): the median over the 27 runs of a seed, and in brackets the lowest and highest.",
              "The runs rank the same images, so they are replicates of one estimate, not votes. A covariate is named",
              f"only if its median partial |Spearman| is at least {MIN_RHO:.2f} in both seeds and larger than every",
              "other covariate's (NOTES.md #15). Grain's horizontal and vertical parts are shown alone, as a check."]
    for split in ("test", "val"):
        for name in LABELS:
            cells = [row for row in correlations if row["split"] == split and row["class"] == name]
            named = named_covariate(cells)
            lines += ["", f"**{'Test' if split == 'test' else 'Validation'}, {name} images** - named: "
                          f"{dict(COVARIATES)[named] if named else 'none'}", "",
                      "| Covariate | Spearman, seed 42 | Spearman, seed 43 | Partial, seed 42 | Partial, seed 43 |",
                      "|---|---|---|---|---|"]
            for column, label in COVARIATES + DIRECTIONS:
                own = [row for row in cells if row["covariate"] == column]
                if not own:
                    lines.append(f"| {label} | the same for every image | | | |")
                    continue
                parts = [median_line([row[measure] for row in own if row["seed"] == seed])
                         for measure in ("spearman", "partial") for seed in SEEDS]
                lines.append(f"| {label} | " + " | ".join(parts) + " |")
    lines += ["", "## The #33 link: the healthy test images' margin increase from 20% to 100% (descriptive)", "",
              "Per architecture, regime and seed, the Spearman correlation of each covariate with how much a healthy",
              "test image's margin rose from the 20% run to the 100% run: median over the 9 pairs of a seed.", "",
              "| Covariate | Seed 42 | Seed 43 |", "|---|---|---|"]
    for column, label in COVARIATES:
        parts = [median_line([row["spearman"] for row in shifts if row["covariate"] == column and row["seed"] == seed])
                 for seed in SEEDS]
        lines.append(f"| {label} | " + " | ".join(parts) + " |")
    lines += ["", "## Where the true-class maps sit (468 subset images per run)", "",
              "Background cells are the 7x7 cells more than half of whose pixels are dark; edge cells are the top",
              "corners and the bottom row. Median over the 18 runs of an architecture (both seeds), empty maps left",
              "out and counted.", ""]
    lines += region_lines(per_run, comparisons)
    return "\n".join(lines) + "\n"


def region_lines(per_run, comparisons):
    """Return the regions' markdown: the median of each measure per architecture and class, then the paired
    comparisons - the same layout for the true-class maps and the margin maps."""
    lines = ["| Architecture | Class | Share on background cells | Share on top corners and bottom row | "
             "Peak on a background cell | Empty maps |", "|---|---|---|---|---|---|"]
    for model in MODELS:
        for name in LABELS:
            own = [row for row in per_run if row["model"] == LOG_LABELS[model] and row["class"] == name]
            lines.append(f"| {LOG_LABELS[model]} | {name} | "
                         + " | ".join(f"{np.median([row[measure] for row in own]):.3f}"
                                      for measure, _, _ in REGION_MEASURES)
                         + f" | {sum(row['empty'] for row in own)} |")
    lines += ["", "**Paired comparisons** - second run minus first, per class; regime rows over 9 cells (architecture",
              "and data size), data-size rows over 9 cells (architecture and regime):", ""]
    groups = [(f"{contrast}, {name}", {"kind": kind, "contrast": contrast, "class": name})
              for kind, contrasts in (("regime", ("mild-none", "strong-none", "strong-mild")),
                                      ("data size", ("50%-20%", "100%-20%")))
              for contrast in contrasts for name in LABELS]
    return lines + direction_lines(comparisons, groups, REGION_MEASURES)


def availability_figure(table, path):
    """Draw each covariate's ROC-AUC three ways, with 0.5 - no separation - marked."""
    figure = Figure(figsize=(8.4, 4.6), layout="constrained")
    panel = figure.subplots()
    positions = np.arange(len(table))[::-1]
    for offset, (column, label, colour) in enumerate((("classes_training", "classes, training pool", "#0072b2"),
                                                     ("classes_test", "classes, test", "#d55e00"),
                                                     ("healthy_shift", "healthy, test vs training", "#009e73"))):
        panel.scatter([row[column] for row in table], positions + (1 - offset) * 0.22, color=colour, s=28, label=label)
    panel.axvline(0.5, color="#6b6b68", linewidth=1)
    panel.set_yticks(positions, [row["label"] for row in table], fontsize=8)
    # A little room past 0 and 1, so a dot at a perfect separation is not cut in half by the frame.
    panel.set_xlim(-0.02, 1.02)
    panel.set_xlabel("ROC-AUC of the covariate alone (0.5 = no separation)")
    panel.grid(axis="x", alpha=0.25, linewidth=0.6)
    panel.legend(fontsize=7, loc="lower right")
    title = "What the data offers: each image property alone as a classifier - classes, and healthy test vs training"
    titled(figure, title)
    return save_figure(figure, path, title)


def regions_figure(per_run, measure, ylabel, title, path):
    """Draw one region measure of the true-class maps against data size, per architecture and class."""
    sizes = [LOG_LABELS[split] for split in TRAIN_SPLITS]
    figure = Figure(figsize=(9.6, 6.6), layout="constrained")
    grid = figure.subplots(len(LABELS), len(MODELS), sharey=True, squeeze=False)
    for panels, name in zip(grid, LABELS):
        for panel, model in zip(panels, MODELS):
            for regime in AUGMENTATIONS:
                for seed_number, seed in enumerate(SEEDS):
                    own = {row["data_size"]: row for row in per_run if row["seed"] == seed and row["class"] == name
                           and row["model"] == LOG_LABELS[model] and row["augmentation"] == LOG_LABELS[regime]}
                    colour = REGIME_COLOURS[regime]
                    panel.plot(range(len(sizes)), [own[size][measure] for size in sizes], color=colour,
                               marker=REGIME_MARKERS[regime], markersize=5, linestyle="-" if seed_number == 0 else "--",
                               markerfacecolor=colour if seed_number == 0 else "white", label=f"{regime}, seed {seed}")
            panel.set_xticks(range(len(sizes)), sizes)
            panel.set_title(f"{LOG_LABELS[model]} - {name} images", fontsize=9)
            panel.grid(alpha=0.25, linewidth=0.6)
            panel.set_axisbelow(True)
        panels[0].set_ylabel(ylabel, fontsize=8)
    for panel in grid[-1]:
        panel.set_xlabel("training data")
    figure.legend(*grid[0][0].get_legend_handles_labels(), ncols=6, fontsize=7, frameon=False,
                  loc="outside lower center")
    titled(figure, title)
    return save_figure(figure, path, title)


def analyse(root=None):
    """Compute the covariates, regions and intervention readings into results/shortcut/; return the summary text."""
    rows = covariates(root)
    # The margins are joined to the covariates by "CLASS/filename", so that name must be unique in a split.
    for split in SPLIT_FILES:
        images = [row["image"] for row in rows if row["split"] == split]
        if len(set(images)) != len(images):
            raise ValueError(f"image names repeat within {split}; the join to the margins would be ambiguous")
    runs = load_runs(root)
    first = next(iter(runs.values()))
    names = [subset_name(record) for record in first["records"]]
    weights = subset_weights(names)
    table = availability(rows)
    correlations = margin_correlations(runs, rows)
    shifts = shift_correlations(runs, rows)
    arrays = region_arrays(runs, rows, root)
    per_run = region_rows(runs, arrays, weights)
    comparisons = region_comparisons(arrays, weights)
    difference = preprocessing_check()

    # The intervention. margin_maps.npy carries no names: intervene wrote it in the subset file's order, and
    # load_runs has already made every run's stability records share one order - so one comparison covers all.
    if image_names(load_split("stability_subset")) != names:
        raise ValueError("the stability subset's order differs from the runs' records; the margin maps cannot be read")
    results = intervention_rows(runs, rows, root)
    readings = intervention_readings(results)
    shares = specificity_shares(results)
    links = encoding_links(results, runs)
    pixels = pixel_changes(root)
    margin_arrays = region_arrays(runs, rows, root, "margin")
    margin_per_run = region_rows(runs, margin_arrays, weights)
    margin_comparisons = region_comparisons(margin_arrays, weights)
    audit_empty = audit_margin_empty(runs, root)
    # The unperturbed test specificity, recounted from the intervention's logits, against every results file.
    specificity = max(abs(1 - row["called_pneumonia_before"] / row["images"]
                          - runs[row["experiment"], row["seed"]]["metrics"]["specificity"])
                      for row in results if row["split"] == "test" and row["class"] == "NORMAL")
    checks = (len({(row["experiment"], row["seed"]) for row in results}), specificity)

    directory = results_directory(root, "shortcut")
    write_csv(table, directory / "availability.csv")
    write_csv(correlations, directory / "correlations.csv")
    write_csv(shifts, directory / "shift_correlations.csv")
    write_csv(per_run, directory / "regions.csv")
    write_csv(comparisons, directory / "region_comparisons.csv")
    write_csv(results, directory / "intervention.csv")
    write_csv(margin_per_run, directory / "margin_regions.csv")
    write_csv(margin_comparisons, directory / "margin_region_comparisons.csv")
    text = summary(rows, table, correlations, shifts, per_run, comparisons, difference)
    text += intervention_summary(rows, results, readings, shares, links, pixels, margin_per_run, margin_comparisons,
                                 audit_empty, checks)
    (directory / "summary.md").write_text(text, encoding="utf-8")
    intervention_figure(results, directory / "shortcut_intervention.png")
    regions_figure(margin_per_run, "edge_share", "share of the margin map\non the top corners and bottom row",
                   "Where the margin Grad-CAM maps sit: their share on the top corners and bottom row, both seeds",
                   directory / "shortcut_margin_regions_edge.png")
    availability_figure(table, directory / "shortcut_availability.png")
    regions_figure(per_run, "background_share", "share of the map\non background cells",
                   "Where the true-class Grad-CAM maps sit: their share on the image's background cells, both seeds",
                   directory / "shortcut_regions_background.png")
    regions_figure(per_run, "edge_share", "share of the map on the top\ncorners and bottom row",
                   "Where the true-class Grad-CAM maps sit: their share on the top corners and bottom row, both seeds",
                   directory / "shortcut_regions_edge.png")
    print(f"  preprocessing against the evaluation transform: largest difference {difference:.1e}")
    print(f"{len(rows)} images, {len(correlations)} correlation rows, {len(per_run)} region rows, {len(results)} "
          f"intervention rows, {len(margin_per_run)} margin-region rows written to {directory}")
    return text


def audit_runs():
    """Return the audit's six runs: each architecture without and with strong augmentation, 100% data, seed 42."""
    ids = experiment_ids()
    return [(ids[model, TRAIN_SPLITS[-1], regime], SEED) for model in MODELS for regime in ("none", "strong")]


def audit_images(names):
    """Return the audit's subset positions: 5 images of each class, drawn by seed, the same for every run."""
    rng = random.Random(f"{SEED}:audit")
    chosen = []
    for name in LABELS:
        positions = [position for position, image in enumerate(names) if image.startswith(f"{name}/")]
        chosen += sorted(rng.sample(positions, AUDIT_PER_CLASS))
    return chosen


def audit_items(names, margin=False):
    """Return the audit's overlays in sheet order, as (run key, subset position, map).

    First the 60 true-class overlays, shuffled; with margin, then the margin maps of the 5 healthy images in
    the 6 runs, shuffled by a generator of their own - so overlays 1-60 are the same with or without them.
    """
    positions = audit_images(names)
    items = [(key, position, "true class") for key in audit_runs() for position in positions]
    # A second seeded generator shuffles the order, so neighbouring overlays are not from one run.
    random.Random(f"{SEED}:audit order").shuffle(items)
    if margin:
        extra = [(key, position, "margin") for key in audit_runs() for position in positions
                 if names[position].startswith("NORMAL/")]
        random.Random(f"{SEED}:audit margin order").shuffle(extra)
        items += extra
    return items


def audit_margin_empty(runs, root=None):
    """Return (empty, total): how many of the audit's margin maps - its healthy images in its runs - are empty."""
    names = [subset_name(record) for record in next(iter(runs.values()))["records"]]
    healthy = [position for position in audit_images(names) if names[position].startswith("NORMAL/")]
    empty = 0
    for key in audit_runs():
        maps = np.load(results_directory(root, "runs") / run_directory(runs[key]["config"]).name / "margin_maps.npy")
        empty += int((maps[healthy].reshape(len(healthy), -1).max(axis=1) <= 0).sum())
    return empty, len(healthy) * len(audit_runs())


def audit_sheets(root=None):
    """Write the audit's numbered overlay sheets, its key and a label template to results/shortcut/audit/.

    Every overlay pairs the image with its map, scaled to the map's own maximum as in the Grad-CAM
    examples figure: 60 true-class maps, then - if at most AUDIT_MAX_EMPTY of them are empty - the 30
    margin maps of the healthy images (NOTES.md #15). Nothing on a sheet names the run, the class or
    the map: which overlay is which is in key.csv, not opened while labelling.
    """
    runs = load_runs(root)
    directory = results_directory(root, "shortcut") / "audit"
    directory.mkdir(exist_ok=True)
    names = [subset_name(record) for record in next(iter(runs.values()))["records"]]
    positions = audit_images(names)
    dataset = ChestXrayDataset(load_split("stability_subset"), unnormalised_transform())
    pictures = {position: dataset[position][0][0].numpy() for position in positions}
    empty, total = audit_margin_empty(runs, root)
    items = audit_items(names, margin=empty <= AUDIT_MAX_EMPTY)
    maps = {}
    for key in audit_runs():
        folder = results_directory(root, "runs") / run_directory(runs[key]["config"]).name
        maps[key, "true class"] = np.load(folder / "stability_maps.npy")[:, 0]
        maps[key, "margin"] = np.load(folder / "margin_maps.npy")
    key_rows = []
    for start in range(0, len(items), AUDIT_PER_SHEET):
        figure = Figure(figsize=(12.8, 5.4), layout="constrained")
        grid = figure.subplots(3, 8)
        for slot, axes in enumerate(grid.reshape(-1, 2)):
            number = start + slot + 1
            for panel in axes:
                panel.set_xticks([])
                panel.set_yticks([])
            if start + slot >= len(items):
                axes[0].set_axis_off()
                axes[1].set_axis_off()
                continue
            key, position, kind = items[start + slot]
            cam = maps[key, kind][position]
            axes[0].imshow(pictures[position], cmap="gray", vmin=0, vmax=1)
            axes[1].imshow(pictures[position], cmap="gray", vmin=0, vmax=1)
            heat = upsample(torch.from_numpy(cam[None]))[0].numpy()
            axes[1].imshow(heat / max(float(cam.max()), 1e-12), cmap="inferno", alpha=0.45, vmin=0, vmax=1)
            axes[0].set_title(f"#{number}", fontsize=9)
            axes[1].set_title("empty map" if cam.max() <= 0 else "", fontsize=7)
            key_rows.append({"number": number, "experiment": key[0], "seed": key[1], "image": names[position],
                             "map": kind})
        title = f"Shortcut audit - overlays {start + 1} to {min(start + AUDIT_PER_SHEET, len(items))}"
        titled(figure, title)
        save_figure(figure, directory / f"sheet_{start // AUDIT_PER_SHEET + 1}.png", title)
    write_csv(key_rows, directory / "key.csv")
    write_csv([{"number": row["number"], "label": "", "second_label": "", "note": ""} for row in key_rows],
              directory / "labels_template.csv")
    (directory / "README.md").write_text(
        "# Shortcut audit (NOTES.md #15)\n\n"
        "Written by `python -m src.shortcut --audit-sheets`. For each numbered overlay, write the main hot spot's\n"
        "place in `labels.csv` (a copy of `labels_template.csv`), one of: " + ", ".join(AUDIT_LABELS) + ". A\n"
        "second label is optional. Label overlays 1-60 first, then any after 60. Do not open `key.csv` while\n"
        "labelling: it says which run, image and map each overlay is.\n", encoding="utf-8")
    print(f"{len(items)} overlays on {len(range(0, len(items), AUDIT_PER_SHEET))} sheets written to {directory} "
          f"({empty} of the {total} margin maps empty; at most {AUDIT_MAX_EMPTY} allowed)")


def reencode(image, quality=QUALITY):
    """Return a grayscale PIL image saved as a JPEG of the given quality in memory, then read back."""
    # BytesIO is an in-memory file: the JPEG is written and read without touching the disk.
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as source:
        return source.convert("L")


def blurred(units):
    """Return 0-1 images blurred as the grain measure blurs them: Gaussian, sigma 1 px, kernel 7."""
    return TF.gaussian_blur(units, kernel_size=[GRAIN_KERNEL, GRAIN_KERNEL], sigma=[GRAIN_SIGMA, GRAIN_SIGMA])


def sharpened(units):
    """Return 0-1 images unsharp-masked - the image plus what the blur removes - clipped back into 0-1.

    Before the clipping, the change from the image is exactly the blur's change with its sign reversed.
    """
    return (2 * units - blurred(units)).clamp(0, 1)


class InterventionImages(Dataset):
    """One split's images in the intervention's four versions: ((4, 224, 224) 0-1 tensor, label) per image.

    Each version is the evaluation preprocessing before normalisation, one grey channel (the three are
    identical copies). The re-encode happens to the grayscale original, before the resize, as a file saved
    at another quality would have been read; blur and sharpen act on the 224x224 image.
    """

    def __init__(self, paths):
        # No transform: the dataset hands over the grayscale PIL image, so it can be re-encoded first.
        self.images = ChestXrayDataset(paths)
        self.transform = unnormalised_transform()

    def __len__(self):
        """Return the number of images."""
        return len(self.images)

    def __getitem__(self, index):
        """Return the four versions of one image and its label."""
        image, label = self.images[index]
        # [:1] keeps the first channel with its axis, (1, 224, 224), so the blur sees one image.
        clean = self.transform(image)[:1]
        return torch.cat([clean, self.transform(reencode(image))[:1], blurred(clean), sharpened(clean)]), label


def intervention_images(split, workers=0):
    """Return (units (N, 4, 224, 224), labels, names) for one split: every image in the four versions.

    Built once per session and shared by every run - evaluation preprocessing is the same for all 27
    configurations - so each image is decoded and re-encoded once, not 54 times. About 1.1 GB for
    validation and test together.
    """
    dataset = InterventionImages(load_split(split))
    loader = DataLoader(dataset, batch_size=RECIPE.batch_size, num_workers=workers)
    units, labels = [], []
    for batch, label in loader:
        units.append(batch)
        labels.append(label)
    return torch.cat(units), torch.cat(labels), image_names(dataset.images.paths)


def normalised(units):
    """Return one version of a batch, (N, 224, 224) in 0-1, as the models' input: 3 channels, normalised."""
    # [:, None] adds the channel axis; expand repeats the grey channel three times without copying.
    return v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(units[:, None].expand(-1, 3, -1, -1))


def pass_logits(model, units, device):
    """Return a model's logits for one version of a split's images, (N, 224, 224), as a CPU tensor.

    Batched exactly as the evaluation batched them - RECIPE.batch_size images, in split order, normalised
    on the CPU - because a convolution's result can depend on the batch it runs in, and the unperturbed
    pass has to reproduce the collected logits.
    """
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(units), RECIPE.batch_size):
            outputs.append(model(normalised(units[start:start + RECIPE.batch_size]).to(device)).cpu())
    return torch.cat(outputs)


def intervention_gate(collected, fresh, threshold):
    """Return (verdict, largest difference): do the unperturbed logits reproduce the collected ones?

    collected and fresh map each split to its logits. "PASS-exact" when every logit is identical;
    "PASS-approx" when all are within GATE_APPROX and no prediction differs, at 0.5 or at the run's
    tuned threshold; "FAIL" otherwise, and that run's perturbed scores are not used.
    """
    largest = max(float((fresh[split] - collected[split]).abs().max()) for split in collected)
    same = all(np.array_equal(classify(fresh[split], cut), classify(collected[split], cut))
               for split in collected for cut in (0.0, threshold))
    if largest == 0:
        return "PASS-exact", largest
    if largest <= GATE_APPROX and same:
        return "PASS-approx", largest
    return "FAIL", largest


def margin_maps(model, name, units, device, batch_size=32):
    """Return the margin Grad-CAM maps (N, 7, 7) of unperturbed images, (N, 224, 224), as float32 NumPy."""
    maps = [gradcam(model, name, normalised(units[start:start + batch_size]).to(device), MARGIN)[0].cpu()
            for start in range(0, len(units), batch_size)]
    return torch.cat(maps).numpy().astype(np.float32)


def intervene(seed=SEED, root=None, workers=0, run_ids=None):
    """Score every experiment of one seed on the four versions of validation and test; return {ID: verdict}.

    Per run, in results/runs/<run>/: margin_maps.npy, the stability subset's margin Grad-CAM maps in the
    subset's order, and shortcut.json - the gate, the versions, and every image's logits under every
    pass on both splits. A run whose shortcut.json exists is skipped, so an interrupted session picks up
    where it stopped; checkpoints, results files and scores.json are only read.
    """
    # get_device also switches cuDNN to deterministic mode, as for the evaluation that is reproduced here.
    device, device_name = get_device()
    pending, verdicts = [], {}
    for run_id in run_ids or EXPERIMENTS:
        name = run_directory(get_config(run_id, seed), root).name
        if (results_directory(root, "runs") / name / "shortcut.json").exists():
            print(f"{name}: already done, skipping")
        else:
            pending.append(run_id)
    if not pending:
        return verdicts
    started = time.perf_counter()
    splits = {split: intervention_images(split, workers) for split in ("val", "test")}
    print(f"  four versions of {sum(len(units) for units, _, _ in splits.values())} images prepared in "
          f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
    test_names = splits["test"][2]
    subset = [test_names.index(name) for name in image_names(load_split("stability_subset"))]
    for run_id in pending:
        started = time.perf_counter()
        config = get_config(run_id, seed)
        run_dir = run_directory(config, root)
        target = results_directory(root, "runs") / run_dir.name
        collected = json.loads((target / "scores.json").read_text(encoding="utf-8"))
        for split, (_, labels, names) in splits.items():
            if names != collected[split]["images"] or labels.tolist() != collected[split]["labels"]:
                raise ValueError(f"{run_dir.name}: the {split} images differ from the collected ones")
        metrics_path = results_directory(root) / f"{output_stem(config)}_metrics.json"
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
        model = load_best(config, run_dir, device)
        logits = {split: {name: pass_logits(model, units[:, index], device) for index, name in enumerate(PASSES)}
                  for split, (units, _, _) in splits.items()}
        verdict, largest = intervention_gate(
            {split: torch.tensor(collected[split]["logits"], dtype=torch.float32) for split in splits},
            {split: logits[split]["unperturbed"] for split in splits}, stored["threshold_logit_margin"])
        maps = margin_maps(model, config.model, splits["test"][0][subset, 0], device)
        empty = int((maps.reshape(len(maps), -1).max(axis=1) <= 0).sum())

        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "margin_maps.npy", maps)
        payload = {
            "experiment_id": run_id, "seed": seed, "gate": verdict, "largest_difference": largest,
            "code_version": code_version(), "device": device_name, "torch": torch.__version__,
            "torchvision": torchvision.__version__, "collected_torch": collected["torch"],
            "passes": list(PASSES), "quality": QUALITY, "blur_sigma": GRAIN_SIGMA, "blur_kernel": GRAIN_KERNEL,
            "margin_maps_empty": empty,
            **{split: {"images": names, "labels": labels.tolist(),
                       "logits": {name: values.tolist() for name, values in logits[split].items()}}
               for split, (_, labels, names) in splits.items()},
        }
        # Written last and by rename: shortcut.json marks a run as done, so it must never exist half written.
        temporary = target / "shortcut.json.tmp"
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(target / "shortcut.json")
        verdicts[run_id] = verdict
        print(f"{run_dir.name}: {verdict} (largest difference {largest:.1e}; torch {torch.__version__} on "
              f"{device_name}, collected with {collected['torch']}); empty margin maps {empty} of {len(maps)}; "
              f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return verdicts


def version_changes(before, after, truth, at_q75):
    """Return {class: measures}: how one version moved the margins of each class's images.

    before and after are (N, 2) logits of the same images, unperturbed and under the version; truth holds their
    labels and at_q75 flags the images whose original is a quality-75 JPEG, both NumPy arrays. delta is the
    median over the class's images of each image's margin change - the one estimator every reading uses. The
    rest is descriptive: the mean change, the change of the class's median margin, delta over the quality-75
    images only, and how many of the class's images are called PNEUMONIA at 0.5 before and after, and flip.
    """
    old, new = margins(before), margins(after)
    called_before, called = classify(before), classify(after)
    change = new - old
    measures = {}
    for name, index in LABELS.items():
        keep = truth == index
        q75 = keep & at_q75
        measures[name] = {
            "images": int(keep.sum()), "delta": float(np.median(change[keep])),
            "mean_change": float(change[keep].mean()),
            "median_shift": float(np.median(new[keep]) - np.median(old[keep])),
            "q75_images": int(q75.sum()),
            "delta_q75": float(np.median(change[q75])) if q75.any() else float("nan"),
            "called_pneumonia_before": int(called_before[keep].sum()), "called_pneumonia": int(called[keep].sum()),
            "to_pneumonia": int((keep & (called == 1) & (called_before == 0)).sum()),
            "to_normal": int((keep & (called == 0) & (called_before == 1)).sum()),
        }
    return measures


def intervention_rows(runs, rows, root=None):
    """Return one row per run, split, class and version - version_changes - read from each run's shortcut.json.

    A run whose intervention failed its gate is left out. For the others the images, labels and unperturbed
    logits must equal the collected ones exactly, and the versions must be the ones this code describes.
    """
    at_q75 = {(row["split"], row["image"]): row["quant"] == Q75_TABLE for row in rows}
    results = []
    for key, run in runs.items():
        folder = results_directory(root, "runs") / run_directory(run["config"]).name
        payload = json.loads((folder / "shortcut.json").read_text(encoding="utf-8"))
        if payload["gate"] == "FAIL":
            print(f"  {folder.name}: the intervention failed its gate, so it is left out")
            continue
        if (payload["quality"], payload["blur_sigma"], payload["blur_kernel"]) != (QUALITY, GRAIN_SIGMA, GRAIN_KERNEL):
            raise ValueError(f"{folder.name}: shortcut.json was made with other versions than this code describes")
        # load_runs keeps the validation scores as val_images, val_labels, val_logits and the test ones unprefixed.
        for split, prefix in (("val", "val_"), ("test", "")):
            record, labels = payload[split], run[f"{prefix}labels"]
            logits = {name: torch.tensor(values, dtype=torch.float32) for name, values in record["logits"].items()}
            if (record["images"] != run[f"{prefix}images"] or record["labels"] != labels.tolist()
                    or not torch.equal(logits["unperturbed"], run[f"{prefix}logits"])):
                raise ValueError(f"{folder.name}: the intervention's {split} images or unperturbed logits differ "
                                 "from the collected ones")
            flags = np.array([at_q75[split, image] for image in record["images"]])
            for version in VERSIONS:
                changes = version_changes(logits["unperturbed"], logits[version], labels.numpy(), flags)
                for name, measures in changes.items():
                    results.append({**run_columns(run), "split": split, "class": name, "version": version,
                                    **measures})
    return results


def intervention_readings(results):
    """Return (verdict, {architecture: counts}) by the readings stated in the plan (NOTES.md #15).

    Per architecture, over its runs: "supported" and "weak" count the encoding's two conditions on the healthy
    validation images' re-encode delta; "sharpen_val" and "sharpen_test" count the healthy images' sharpen
    delta below, above and exactly at 0; "blur_up" counts blur deltas above 0 on validation. Each architecture
    gets the words its counts earn; the verdict is "supported" or "not supported" only if every architecture
    earns it, otherwise "per architecture".
    """
    delta = {(row["experiment"], row["seed"], row["split"], row["class"], row["version"]): row["delta"]
             for row in results}
    counts = {}
    for model in MODELS:
        keys = sorted({(row["experiment"], row["seed"]) for row in results if row["model"] == LOG_LABELS[model]})
        encoding = [delta[key + ("val", "NORMAL", "reencoded")] for key in keys]
        control = [delta[key + ("val", "PNEUMONIA", "reencoded")] for key in keys]
        own = {"runs": len(keys),
               "supported": sum(rise >= ENCODING_RISE and rise >= ENCODING_RATIO * abs(other)
                                for rise, other in zip(encoding, control)),
               "weak": sum(rise < ENCODING_WEAK for rise in encoding),
               "blur_up": sum(delta[key + ("val", "NORMAL", "blur")] > 0 for key in keys)}
        own["encoding"] = ("supported" if own["supported"] >= VERDICT_RUNS
                           else "not supported" if own["weak"] >= VERDICT_RUNS else "neither")
        for split in ("val", "test"):
            sharpen = [delta[key + (split, "NORMAL", "sharpen")] for key in keys]
            below, above = sum(value < 0 for value in sharpen), sum(value > 0 for value in sharpen)
            own[f"sharpen_{split}"] = (below, above, sum(value == 0 for value in sharpen))
            own[f"texture_{split}"] = ("texture" if below >= VERDICT_RUNS
                                       else "anything unusual" if above >= VERDICT_RUNS else "neither")
        counts[LOG_LABELS[model]] = own
    words = [own["encoding"] for own in counts.values()]
    verdict = words[0] if len(set(words)) == 1 and words[0] != "neither" else "per architecture"
    return verdict, counts


def specificity_shares(results):
    """Return per run the specificity at 0.5 on validation before and after the re-encode and on test, and the share.

    The share of the validation-to-test gap the re-encode reproduces is (s_v - s_r) / (s_v - s_t): None when
    the gap is below MIN_GAP, and never clipped. Descriptive only - not a second verdict.
    """
    both = {}
    for row in results:
        if row["class"] == "NORMAL" and row["version"] == "reencoded":
            both.setdefault((row["model"], row["experiment"], row["seed"]), {})[row["split"]] = row
    shares = []
    for (model, experiment, seed), rows in both.items():
        val, test = rows["val"], rows["test"]
        s_v = 1 - val["called_pneumonia_before"] / val["images"]
        s_r = 1 - val["called_pneumonia"] / val["images"]
        s_t = 1 - test["called_pneumonia_before"] / test["images"]
        shares.append({"model": model, "experiment": experiment, "seed": seed, "s_v": s_v, "s_r": s_r, "s_t": s_t,
                       "false_alarms_val": val["called_pneumonia_before"],
                       "false_alarms_reencoded": val["called_pneumonia"],
                       "false_alarms_test": test["called_pneumonia_before"],
                       "share": (s_v - s_r) / (s_v - s_t) if s_v - s_t >= MIN_GAP else None})
    return shares


def encoding_links(results, runs):
    """Return {seed: (across, paired, runs, cells)}: how the re-encode delta goes with the #33 shift, descriptive.

    Across: the Spearman correlation over a seed's runs of the healthy validation images' re-encode delta with
    the run's healthy test images' median margin. Paired, closer to #33: over the 9 architecture x regime cells,
    the change of that delta from 20% to 100% of the data against the change of that median margin.
    """
    ids = experiment_ids()
    delta = {(row["experiment"], row["seed"]): row["delta"] for row in results
             if row["split"] == "val" and row["class"] == "NORMAL" and row["version"] == "reencoded"}
    links = {}
    for seed in SEEDS:
        keys = [key for key in delta if key[1] == seed]
        across = spearman([delta[key] for key in keys], [runs[key]["normal_margin_median_test"] for key in keys])
        cells = [(ids[model, TRAIN_SPLITS[0], regime], ids[model, TRAIN_SPLITS[-1], regime])
                 for model in MODELS for regime in AUGMENTATIONS]
        changes = [(delta[large, seed] - delta[small, seed],
                    runs[large, seed]["normal_margin_median_test"] - runs[small, seed]["normal_margin_median_test"])
                   for small, large in cells if (small, seed) in delta and (large, seed) in delta]
        # zip(*changes) turns the list of pairs into two sequences: the delta changes and the margin changes.
        links[seed] = (across, spearman(*zip(*changes)), len(keys), len(changes))
    return links


def cached_pixel_rows(path, count):
    """Return the rows of a pixel_changes.csv made with this code's versions and complete, or None."""
    if not path.exists():
        return None
    parameters = {"quality": QUALITY, "blur_sigma": GRAIN_SIGMA, "blur_kernel": GRAIN_KERNEL}
    rows = read_rows(path, [f"change_{version}" for version in VERSIONS] + list(parameters))
    if len(rows) == count and all(row[name] == value for row in rows for name, value in parameters.items()):
        return rows
    return None


def pixel_changes(root=None):
    """Return one row per validation and test image: each version's mean absolute pixel change, in 0-1 units.

    Model-free, so computed once here rather than per run on Colab. Kept in results/shortcut/pixel_changes.csv
    with the versions' parameters, and computed again if they or the number of images no longer match.
    """
    path = results_directory(root, "shortcut") / "pixel_changes.csv"
    count = sum(len(load_split(split)) for split in ("val", "test"))
    rows = cached_pixel_rows(path, count)
    if rows is not None:
        return rows
    rows = []
    for split in ("val", "test"):
        dataset = InterventionImages(load_split(split))
        # iter() makes a one-way cursor over the names; next() takes the next one for each image in turn.
        names = iter(image_names(dataset.images.paths))
        # One batch at a time: all four versions of every image at once would take about 1.1 GB.
        for batch, _ in DataLoader(dataset, batch_size=RECIPE.batch_size):
            # Each version minus the unperturbed one, averaged over the image's pixels: (batch, 3).
            changes = (batch[:, 1:] - batch[:, :1]).abs().flatten(2).mean(dim=2)
            for values in changes:
                rows.append({"split": split, "image": next(names),
                             **{f"change_{version}": float(value) for version, value in zip(VERSIONS, values)},
                             "quality": QUALITY, "blur_sigma": GRAIN_SIGMA, "blur_kernel": GRAIN_KERNEL})
    write_csv(rows, path)
    return rows


def intervention_summary(rows, results, readings, shares, links, pixels, margin_per_run, margin_comparisons,
                         audit_empty, checks):
    """Return the summary.md sections of part 3: the intervention's checks, readings and descriptions, where the
    margin maps sit, and each version's pixel change."""
    verdict, counts = readings
    runs_checked, specificity = checks
    models = [LOG_LABELS[model] for model in MODELS]
    groups = [(split, name) for split in ("val", "test") for name in LABELS]
    split_names = {"val": "validation", "test": "test"}
    quality_75 = {row["image"] for row in rows if row["split"] in split_names and row["quant"] == Q75_TABLE}

    def values_of(model, split, name, version, column="delta"):
        """Return one column of an architecture's rows for a split, class and version."""
        return [row[column] for row in results if row["model"] == model and row["split"] == split
                and row["class"] == name and row["version"] == version]

    def median_of(model, split, name, version, column="delta"):
        """Return the median of one column over an architecture's runs."""
        return float(np.median(values_of(model, split, name, version, column)))

    def pixel(split, name, version, only_q75=False):
        """Return the median pixel change of one version over a split's images of one class."""
        return float(np.median([row[f"change_{version}"] for row in pixels if row["split"] == split
                                and row["image"].startswith(f"{name}/")
                                and (not only_q75 or row["image"] in quality_75)]))

    lines = ["", "## The intervention (part 3)", "",
             "Every checkpoint scored validation and test again on the Colab T4 that collected the scores",
             "(`python -m src.shortcut --intervene`, `results/runs/*/shortcut.json`) in four versions: unperturbed,",
             f"re-encoded as a JPEG of quality {QUALITY} (the healthy test images' encoding), blurred (Gaussian, sigma "
             f"{GRAIN_SIGMA:g} px) and",
             "sharpened (the image plus what the blur removes: the blur's pixel change before the clip to 0-1, less",
             "after it). *delta* is the median over a class's images of each image's margin change against the",
             "unperturbed version, in logits; the margin is the PNEUMONIA logit minus the NORMAL logit, and above 0 is",
             "called pneumonia at 0.5.", "", "**Checks:**", "",
             f"- The images, labels and unperturbed logits of all {runs_checked} runs equal the collected ones exactly.",
             "- The margin maps are read in the stability subset's order, which equals every run's stability records.",
             f"- Test specificity at 0.5 from the unperturbed logits against every results file: largest difference "
             f"{specificity:.1e}.", "",
             "**Encodings in the data** (`covariates.csv`; a luminance-table mean of 29.03 is quality 75, a lower mean a",
             "higher quality):", "",
             "| Split, class | Images | Quality 75 | Higher quality | Lower quality |", "|---|---|---|---|---|"]
    for split, name in groups:
        quants = [row["quant"] for row in rows if row["split"] == split and row["class"] == name]
        lines.append(f"| {split_names[split]}, {name} | {len(quants)} | {sum(q == Q75_TABLE for q in quants)} | "
                     f"{sum(q < Q75_TABLE for q in quants)} | {sum(q > Q75_TABLE for q in quants)} |")

    lines += ["", "### The readings, stated before any perturbed score was read (NOTES.md #15)", "",
              f"- **Encoding:** supported if, on validation, delta(NORMAL) >= {ENCODING_RISE:g} logit and >= "
              f"{ENCODING_RATIO:g} x |delta(PNEUMONIA)| in at least {VERDICT_RUNS} of the 18 runs of every "
              f"architecture; not supported if delta(NORMAL) < {ENCODING_WEAK:g} in at least {VERDICT_RUNS} of the "
              "18 runs of every architecture; otherwise stated per architecture.",
              f"- **Texture or anything unusual:** sharpening moves the healthy validation images towards NORMAL "
              f"(delta < 0) in at least {VERDICT_RUNS} of an architecture's 18 runs - texture; towards PNEUMONIA "
              f"(delta > 0) in at least {VERDICT_RUNS} - anything unusual looks like pneumonia; otherwise neither. A "
              "delta of exactly 0 counts for neither. Read on validation, test beside it (decided on 2026-09-28, "
              "before any number was seen); a different word on test is a disagreement.",
              "- An architecture's 18 runs are 9 configurations x 2 seeds; the two seeds of a configuration are not "
              "independent, so a count is not 18 independent votes.", "",
              f"**Encoding: {verdict}**"
              + ("." if verdict != "per architecture"
                 else " - " + "; ".join(f"{model}: {counts[model]['encoding']}" for model in models) + "."), "",
              "| Architecture | Runs | Encoding: >= 1 and >= 5x | Encoding: < 0.5 | Encoding | Sharpen, validation: "
              "< 0 / > 0 / = 0 | Validation | Sharpen, test: < 0 / > 0 / = 0 | Test | Blur, validation: > 0 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for model in models:
        own = counts[model]
        lines.append(f"| {model} | {own['runs']} | {own['supported']} | {own['weak']} | {own['encoding']} | "
                     + " / ".join(map(str, own["sharpen_val"])) + f" | {own['texture_val']} | "
                     + " / ".join(map(str, own["sharpen_test"])) + f" | {own['texture_test']} | {own['blur_up']} |")
    different = [model for model in models if counts[model]["texture_val"] != counts[model]["texture_test"]]
    lines += ["", "Sharpen, test against validation: "
              + (f"a different verdict for {', '.join(different)}." if different
                 else "the same verdict for every architecture.")
              + f" Pixel change of the healthy validation images: blur {pixel('val', 'NORMAL', 'blur'):.4f}, sharpen "
                f"{pixel('val', 'NORMAL', 'sharpen'):.4f} (equal before the clip)."]

    lines += ["", "**delta per architecture** - median over its runs (lowest to highest):", "",
              "| Version | Split, class | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    for version in VERSIONS:
        for split, name in groups:
            lines.append(f"| {version} | {split_names[split]}, {name} | "
                         + " | ".join(median_line(values_of(model, split, name, version)) for model in models) + " |")

    moved = {version: pixel("val", "NORMAL", version) for version in VERSIONS}
    lines += ["", "**What the re-encode alone cannot separate.** The encoding and \"anything unusual looks like "
              "pneumonia\" both predict that re-encoding raises the healthy margins; how far each version moves the "
              "pixels tells them apart. Healthy validation images: median delta over an architecture's runs / median "
              "pixel change (0-1 units) = delta per unit of pixel change:", "",
              "| Architecture | " + " | ".join(VERSIONS) + " |", "|---|" + "---|" * len(VERSIONS)]
    for model in models:
        lines.append(f"| {model} | " + " | ".join(
            f"{median_of(model, 'val', 'NORMAL', version):+.2f} / {moved[version]:.4f} = "
            f"{median_of(model, 'val', 'NORMAL', version) / moved[version]:+.0f}" for version in VERSIONS) + " |")
    lines.append("")
    for model in models:
        rise, soft = median_of(model, "val", "NORMAL", "reencoded"), median_of(model, "val", "NORMAL", "blur")
        fewer, further = moved["reencoded"] < moved["blur"], abs(rise) > abs(soft)
        lines.append(f"- {model}: the re-encode moves the healthy validation pixels {'less' if fewer else 'more'} "
                     f"than the blur ({moved['reencoded']:.4f} against {moved['blur']:.4f}) and their margins "
                     f"{'more' if further else 'less'} ({rise:+.2f} against {soft:+.2f}) - "
                     + ("specific to the encoding." if fewer and further
                        else "on this comparison, not told apart from a response to any change."))

    lines += ["", "**How much of the validation-to-test gap in specificity the re-encode reproduces** - descriptive, "
              "not a second verdict: a re-encode that moves the healthy margins by 1 logit or more need not carry "
              "them across the cut-off when they sit far below it. Specificity at 0.5 on validation (s_v), after the "
              "re-encode (s_r) and on test (s_t); false alarms among the healthy images; the share is (s_v - s_r) / "
              f"(s_v - s_t), not clipped, and '-' when s_v - s_t < {MIN_GAP:g}. Median over an architecture's runs "
              "(the share: lowest to highest):", "",
              "| Architecture | s_v | s_r | s_t | False alarms: validation / re-encoded / test | Share | Runs with a "
              "share |", "|---|---|---|---|---|---|---|"]
    for model in models:
        own = [row for row in shares if row["model"] == model]
        valid = [row["share"] for row in own if row["share"] is not None]
        alarms = " / ".join(f"{np.median([row[column] for row in own]):.0f}"
                            for column in ("false_alarms_val", "false_alarms_reencoded", "false_alarms_test"))
        lines.append(f"| {model} | " + " | ".join(f"{np.median([row[column] for row in own]):.3f}"
                                                  for column in ("s_v", "s_r", "s_t"))
                     + f" | {alarms} | {median_line(valid)} | {len(valid)} of {len(own)} |")

    lines += ["", "**The test re-encode, a sanity check.** Every healthy test image is already quality 75, so "
              "re-encoding changes it little; re-saving a quality-75 file at 75 still quantises it again, so it is "
              "not a no-op - read beside its pixel change. Median delta over an architecture's runs:", "",
              "| | Validation, NORMAL | Test, NORMAL | Test, PNEUMONIA | Test, PNEUMONIA at quality 75 |",
              "|---|---|---|---|---|"]
    for model in models:
        lines.append(f"| {model} | {median_of(model, 'val', 'NORMAL', 'reencoded'):+.3f} | "
                     f"{median_of(model, 'test', 'NORMAL', 'reencoded'):+.3f} | "
                     f"{median_of(model, 'test', 'PNEUMONIA', 'reencoded'):+.3f} | "
                     f"{median_of(model, 'test', 'PNEUMONIA', 'reencoded', 'delta_q75'):+.3f} |")
    lines.append(f"| Pixel change | {pixel('val', 'NORMAL', 'reencoded'):.4f} | "
                 f"{pixel('test', 'NORMAL', 'reencoded'):.4f} | {pixel('test', 'PNEUMONIA', 'reencoded'):.4f} | "
                 f"{pixel('test', 'PNEUMONIA', 'reencoded', only_q75=True):.4f} |")

    sizes = [LOG_LABELS[split] for split in TRAIN_SPLITS]
    lookup = {(row["model"], row["augmentation"], row["data_size"], row["seed"]): row["delta"] for row in results
              if row["split"] == "val" and row["class"] == "NORMAL" and row["version"] == "reencoded"}
    lines += ["", "**The healthy validation images' re-encode delta by regime and data size** (descriptive) - "
              "seed 42 / seed 43:", "",
              "| Architecture | Regime | " + " | ".join(sizes) + " |", "|---|---|" + "---|" * len(sizes)]
    for model in models:
        for regime in AUGMENTATIONS:
            label = LOG_LABELS[regime]
            lines.append(f"| {model} | {label} | " + " | ".join(
                " / ".join(f"{lookup[model, label, size, seed]:+.2f}" if (model, label, size, seed) in lookup
                           else "-" for seed in SEEDS) for size in sizes) + " |")
    lines += ["", "**The #33 link** (descriptive): the Spearman correlation of the healthy validation images' "
              "re-encode delta with the run's healthy test images' median margin, over a seed's runs; and, closer to "
              "#33, of the change of that delta from 20% to 100% with the change of that margin, over the "
              "architecture x regime cells.", "",
              "| Seed | Across the runs | Change from 20% to 100% |", "|---|---|---|"]
    for seed, (across, paired, run_count, cell_count) in links.items():
        lines.append(f"| {seed} | {across:+.2f} ({run_count} runs) | {paired:+.2f} ({cell_count} cells) |")

    empty, total = audit_empty
    lines += ["", "## Where the margin maps sit (468 subset images per run)", "",
              "A margin map explains the PNEUMONIA logit minus the NORMAL logit, so on a healthy image it shows the",
              "evidence for pneumonia - where a false alarm's evidence sits. Measures as for the true-class maps; median",
              "over the 18 runs of an architecture (both seeds), empty maps left out and counted.", ""]
    lines += region_lines(margin_per_run, margin_comparisons)
    lines += ["", f"**The audit's margin overlays:** {empty} of the {total} margin maps of its healthy images are "
              f"empty; at most {AUDIT_MAX_EMPTY} are allowed (fewer than a quarter), so "
              + ("overlays 61-90 are drawn." if empty <= AUDIT_MAX_EMPTY else "no margin overlays are drawn.")]

    lines += ["", "## Pixel change per version", "",
              "The mean absolute change of each version against the unperturbed image at 224x224, in 0-1 units",
              "(model-free, `pixel_changes.csv`); median over the images. Blur / sharpen is 1 before the clip to 0-1.", "",
              "| Split, class | Re-encoded | Blur | Sharpen | Blur / sharpen |", "|---|---|---|---|---|"]
    for split, name in groups:
        changes = [pixel(split, name, version) for version in VERSIONS]
        lines.append(f"| {split_names[split]}, {name} | " + " | ".join(f"{value:.4f}" for value in changes)
                     + f" | {changes[1] / changes[2]:.2f} |")
    return "\n".join(lines) + "\n"


def intervention_figure(results, path):
    """Draw every run's delta per version, split, class and architecture, with 0 and the encoding thresholds."""
    groups = [("val", "NORMAL"), ("val", "PNEUMONIA"), ("test", "NORMAL"), ("test", "PNEUMONIA")]
    figure = Figure(figsize=(10.8, 8.4), layout="constrained")
    grid = figure.subplots(len(VERSIONS), len(MODELS), sharey="row", squeeze=False)
    # Each regime's dots sit a little left or right of their group, so the three do not overlap.
    offsets = {regime: (index - 1) * 0.22 for index, regime in enumerate(AUGMENTATIONS)}
    for panels, version in zip(grid, VERSIONS):
        for panel, model in zip(panels, MODELS):
            for position, (split, name) in enumerate(groups):
                for regime in AUGMENTATIONS:
                    for seed_number, seed in enumerate(SEEDS):
                        values = [row["delta"] for row in results
                                  if row["model"] == LOG_LABELS[model] and row["augmentation"] == LOG_LABELS[regime]
                                  and row["seed"] == seed and row["split"] == split and row["class"] == name
                                  and row["version"] == version]
                        colour = REGIME_COLOURS[regime]
                        panel.plot([position + offsets[regime]] * len(values), values, linestyle="none",
                                   marker=REGIME_MARKERS[regime], markersize=4.5, color=colour,
                                   markerfacecolor=colour if seed_number == 0 else "white",
                                   label=f"{regime}, seed {seed}" if position == 0 else None)
            panel.axhline(0, color="#6b6b68", linewidth=1)
            if version == "reencoded":
                panel.axhline(ENCODING_WEAK, color="#6b6b68", linewidth=0.8, linestyle="--")
                panel.axhline(ENCODING_RISE, color="#6b6b68", linewidth=0.8, linestyle=":")
            panel.set_xticks(range(len(groups)), [f"{'val' if split == 'val' else 'test'}\n{name}"
                                                  for split, name in groups], fontsize=7)
            panel.set_title(f"{LOG_LABELS[model]} - {version}", fontsize=9)
            panel.grid(axis="y", alpha=0.25, linewidth=0.6)
            panel.set_axisbelow(True)
        panels[0].set_ylabel("delta: median margin\nchange (logits)", fontsize=8)
    figure.legend(*grid[0][0].get_legend_handles_labels(), ncols=6, fontsize=7, frameon=False,
                  loc="outside lower center")
    title = ("The intervention: each run's median margin change per version, split and class (dashed 0.5, dotted 1 "
             "logit: the encoding thresholds)")
    titled(figure, title)
    return save_figure(figure, path, title)


def self_check():
    """Check the regions, the image features, the partial correlation, the audit draw, the intervention's
    versions and gate, and step 3's readings, shares, margin maps and caches on synthetic data."""
    maps = np.zeros((3, CELLS, CELLS))
    maps[0, 0, 0] = 1.0
    maps[1, 3, 3] = 2.0
    background = np.zeros((3, CELLS, CELLS), dtype=bool)
    background[1, 3, 3] = True
    share, edge, category = map_regions(maps, background)
    print(f"  a map in a top corner: edge share 1, background 0, peak on an edge cell: "
          f"{edge[0] == 1 and share[0] == 0 and category[0] == 'edge'}; a map on a background cell: share 1, peak "
          f"there: {share[1] == 1 and category[1] == 'background'}; an empty map: no shares, counted as empty: "
          f"{np.isnan(share[2]) and np.isnan(edge[2]) and category[2] == 'empty'}")

    flat = image_features(torch.full((IMAGE_SIZE, IMAGE_SIZE), 0.5))
    black = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
    print(f"  a flat image has no grain or contrast: {flat['grain'] == 0 and flat['contrast'] == 0}; a black image "
          f"is all dark and all background: {image_features(black)['dark'] == 1 and background_cells(black).all()}")
    # White noise of known sigma: the blur's residual keeps sigma * sqrt(sum((delta - kernel)^2)) of it.
    sigma = 0.02
    noisy = 0.5 + sigma * torch.randn(IMAGE_SIZE, IMAGE_SIZE, generator=torch.Generator().manual_seed(0))
    line = torch.exp(-torch.arange(-3, 4, dtype=torch.float64) ** 2 / (2 * GRAIN_SIGMA ** 2))
    kernel = torch.outer(line / line.sum(), line / line.sum())
    expected = sigma * float(torch.sqrt(1 - 2 * kernel[3, 3] + (kernel ** 2).sum()))
    measured = image_features(noisy)["grain"]
    print(f"  noise of sigma {sigma} gives the expected grain ({measured:.5f} against {expected:.5f}): "
          f"{abs(measured / expected - 1) < 0.03}")

    rng = np.random.default_rng(0)
    common = rng.normal(size=400)
    a, b = common + 0.3 * rng.normal(size=400), common + 0.3 * rng.normal(size=400)
    target = a + 0.3 * rng.normal(size=400)
    alone, given = spearman(b, target), partial_spearman(target, [a, b], 1)
    print(f"  a covariate that only shares a confounder: Spearman {alone:+.2f} alone, {given:+.2f} given the other: "
          f"{alone > 0.5 and abs(given) < 0.15}; the one that matters stays: {partial_spearman(target, [a, b], 0) > 0.5}")

    def cell(covariate, seed, value):
        """Return one made-up correlation row."""
        return {"covariate": covariate, "seed": seed, "partial": value}

    strong = [cell("grain", seed, 0.4) for seed in SEEDS for _ in range(3)]
    strong += [cell("mean", seed, 0.1) for seed in SEEDS for _ in range(3)]
    weak = [dict(row, partial=row["partial"] / 4) for row in strong]
    print(f"  the naming rule: a strong covariate is named ({named_covariate(strong)}), a weak one is not "
          f"({named_covariate(weak)}): {named_covariate(strong) == 'grain' and named_covariate(weak) is None}")

    # A 2-D uint8 array becomes a single-channel "L" image, like the dataset's.
    picture = Image.fromarray(rng.integers(0, 256, size=(300, 420), dtype=np.uint8))
    normalise = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    same = torch.equal(normalise(unnormalised_transform()(picture)), build_transform("none", train=False)(picture))
    names = [f"NORMAL/n{i}.jpeg" for i in range(20)] + [f"PNEUMONIA/p{i}.jpeg" for i in range(20)]
    drawn = audit_images(names)
    print(f"  the preprocessing, normalised, is the evaluation transform exactly: {same}; the audit draws 5 of each "
          f"class, the same every time: {len(drawn) == 10 and drawn == audit_images(names)}")

    # The intervention. 29.03125 is the luminance-table mean of every healthy test image (covariates.csv).
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=QUALITY)
    buffer.seek(0)
    with Image.open(buffer) as saved:
        table = float(np.mean(saved.quantization[0]))
    again = reencode(picture)
    lossy = again.size == picture.size and again.mode == "L" and not np.array_equal(np.asarray(again), np.asarray(picture))
    print(f"  a JPEG at quality {QUALITY} has the healthy test images' table (mean {table:.3f}): "
          f"{table == 29.03125}; the re-encode keeps size and mode and changes pixels: {lossy}")
    softer, sharper = blurred(noisy[None])[0], sharpened(noisy[None])[0]
    grains = [image_features(image)["grain"] for image in (softer, noisy, sharper)]
    changes = [float((image - noisy).abs().mean()) for image in (softer, sharper)]
    print(f"  blur lowers the grain and sharpen raises it ({grains[0]:.4f} < {grains[1]:.4f} < {grains[2]:.4f}): "
          f"{grains[0] < grains[1] < grains[2]}; both change the pixels by the same amount ({changes[0]:.5f}, "
          f"{changes[1]:.5f}): {abs(changes[0] / changes[1] - 1) < 1e-4}")

    # The four versions of a real file, as InterventionImages reads them.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "NORMAL" / "synthetic.jpeg"
        path.parent.mkdir()
        picture.save(path, quality=95)
        versions, label = InterventionImages([path])[0]
        with Image.open(path) as source:
            reference = build_transform("none", train=False)(source.convert("L"))
    exact = torch.equal(normalised(versions[None, 0])[0], reference)
    derived = torch.equal(versions[2:], torch.cat([blurred(versions[:1]), sharpened(versions[:1])]))
    print(f"  four versions {tuple(versions.shape)}, label {label}; the unperturbed one as model input is the "
          f"evaluation's exactly: {exact}; blur and sharpen are of that image: {derived}")

    # The gate on made-up logits: image 0 of validation sits 1e-6 above the cut-off.
    base = {"val": torch.randn(40, 2, generator=torch.Generator().manual_seed(1)),
            "test": torch.randn(30, 2, generator=torch.Generator().manual_seed(2))}
    base["val"][0] = torch.tensor([0.0, 1e-6])
    flipped = {split: values.clone() for split, values in base.items()}
    flipped["val"][0, 1] = -1e-6
    verdicts = [intervention_gate(base, fresh, 0.5)[0]
                for fresh in (base, {split: values + 1e-6 for split, values in base.items()}, flipped,
                              {split: values + 1e-3 for split, values in base.items()})]
    print(f"  the gate: identical logits pass exactly, a 1e-6 drift approximately, a changed prediction or a "
          f"1e-3 drift fails: {verdicts == ['PASS-exact', 'PASS-approx', 'FAIL', 'FAIL']}")

    # Step 3. Raising the PNEUMONIA logit of every healthy image by 2 moves their margins by 2, the others' by 0.
    truth = np.array([0] * 6 + [1] * 4)
    before = torch.randn(10, 2, generator=torch.Generator().manual_seed(3))
    after = before.clone()
    after[:6, 1] += 2
    changes = version_changes(before, after, truth, np.zeros(10, dtype=bool))
    print(f"  a known margin shift of +2 on the healthy images gives delta 2 there and 0 for the others: "
          f"{abs(changes['NORMAL']['delta'] - 2) < 1e-6 and changes['PNEUMONIA']['delta'] == 0}")

    def readings_for(encoding, control, sharpen):
        """Return the readings of made-up runs with these healthy / pneumonia re-encode and sharpen deltas."""
        made = []
        for model in MODELS:
            for number, (rise, other, sharp) in enumerate(zip(encoding, control, sharpen)):
                for split, name, version, value in (("val", "NORMAL", "reencoded", rise),
                                                    ("val", "PNEUMONIA", "reencoded", other),
                                                    ("val", "NORMAL", "blur", 0.1), ("val", "NORMAL", "sharpen", sharp),
                                                    ("test", "NORMAL", "sharpen", sharp)):
                    made.append({"model": LOG_LABELS[model], "experiment": f"{model}-{number}", "seed": SEED,
                                 "split": split, "class": name, "version": version, "delta": value})
        return intervention_readings(made)

    first = LOG_LABELS[MODELS[0]]
    exact = readings_for([1.0] * 12 + [0.7] * 6, [0.2] * 18, [-0.1] * 11 + [0.0] + [0.1] * 6)
    short = readings_for([1.0] * 11 + [0.7] * 7, [0.0] * 18, [-0.1] * 12 + [0.1] * 6)
    ratio = readings_for([1.0] * 18, [0.3] * 18, [0.1] * 18)
    weak = readings_for([0.4] * 12 + [0.7] * 6, [0.0] * 18, [0.0] * 18)
    print(f"  the readings: 12 runs at exactly 1 logit and 5x are supported: {exact[0] == 'supported'}; 11 are not "
          f"(per architecture): {short[0] == 'per architecture' and short[1][first]['encoding'] == 'neither'}; the "
          f"5x condition failing alone: {ratio[1][first]['supported'] == 0}; 12 runs under 0.5 are not supported: "
          f"{weak[0] == 'not supported'}; sharpen 11 below / 6 above / 1 at 0 is neither, texture at 12 below, "
          f"anything unusual at 12 above: {exact[1][first]['sharpen_val'] == (11, 6, 1)} "
          f"{exact[1][first]['texture_val'] == 'neither' and short[1][first]['texture_val'] == 'texture'} "
          f"{ratio[1][first]['texture_val'] == 'anything unusual'}")

    def healthy(split, before, after):
        """Return one made-up healthy re-encode row of 200 images: how many were called PNEUMONIA."""
        return {"model": first, "experiment": "E01", "seed": SEED, "split": split, "class": "NORMAL",
                "version": "reencoded", "images": 200, "called_pneumonia_before": before, "called_pneumonia": after}

    shares = [specificity_shares([healthy("val", 0, after), healthy("test", test, test)])[0]["share"]
              for after, test in ((20, 100), (120, 100), (20, 5))]
    print(f"  the specificity share: (1 - 0.9) / (1 - 0.5) = 0.2: {abs(shares[0] - 0.2) < 1e-12}; not clipped above "
          f"1: {abs(shares[1] - 1.2) < 1e-12}; '-' for a gap under {MIN_GAP:g}: {shares[2] is None}")

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        config = get_config("E01", SEED)
        target = results_directory(root, "runs") / run_directory(config).name
        target.mkdir()
        margin, true_class = np.zeros((1, CELLS, CELLS), np.float32), np.zeros((1, 5, CELLS, CELLS), np.float32)
        margin[0, 0, 0], true_class[0, 0, 3, 3] = 1.0, 1.0
        np.save(target / "margin_maps.npy", margin)
        np.save(target / "stability_maps.npy", true_class)
        made_runs = {("E01", SEED): {"config": config, "records": [{"class": "NORMAL", "image": "a.jpeg"}]}}
        made_rows = [{"split": "test", "image": "NORMAL/a.jpeg", "background": "0" * CELLS ** 2}]
        margin_edge = region_arrays(made_runs, made_rows, root, "margin")["E01", SEED]["edge_share"][0]
        true_edge = region_arrays(made_runs, made_rows, root)["E01", SEED]["edge_share"][0]
        path = root / "pixel_changes.csv"
        cached = {"split": "val", "image": "NORMAL/a.jpeg", **{f"change_{version}": 0.1 for version in VERSIONS},
                  "quality": QUALITY, "blur_sigma": GRAIN_SIGMA, "blur_kernel": GRAIN_KERNEL}
        write_csv([dict(cached, quality=QUALITY + 5)], path)
        stale = cached_pixel_rows(path, 1) is None
        write_csv([cached], path)
        fresh = cached_pixel_rows(path, 1) is not None and cached_pixel_rows(path, 2) is None
    print(f"  margin maps read whole, true-class maps as pass 0: {margin_edge == 1 and true_edge == 0}; the pixel "
          f"cache refused with other versions or another image count, used otherwise: {stale and fresh}")
    items = audit_items(names, margin=True)
    print(f"  the audit's margin overlays are appended after the 60 true-class ones, which do not change: "
          f"{items[:60] == audit_items(names) and len(items) == 90} "
          f"{all(kind == 'margin' and names[position].startswith('NORMAL/') for _, position, kind in items[60:])}")


def main():
    """Run the local parts, write the audit sheets, run the intervention on Colab, or self-check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit-sheets", action="store_true", help="write the audit's overlay sheets and templates")
    parser.add_argument("--intervene", action="store_true",
                        help="score every experiment on the four versions and save margin maps (GPU)")
    parser.add_argument("--seed", type=int, default=SEED, help=f"which sweep to intervene on (default {SEED})")
    parser.add_argument("--id", action="append", help="intervene on this experiment only; repeatable")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (default: 0)")
    parser.add_argument("--check", action="store_true", help="self-check on synthetic data")
    parser.add_argument("--output-root", type=Path, default=None, help="where checkpoints/ and results/ live")
    args = parser.parse_args()
    if args.check:
        self_check()
    elif args.audit_sheets:
        audit_sheets(args.output_root)
    elif args.intervene:
        verdicts = list(intervene(args.seed, args.output_root, args.workers, args.id).values())
        print(f"seed {args.seed}: " + ", ".join(f"{verdicts.count(kind)} {kind}"
                                                 for kind in ("PASS-exact", "PASS-approx", "FAIL")))
    else:
        analyse(args.output_root)


if __name__ == "__main__":
    main()
