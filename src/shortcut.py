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
* intervention (Colab, GPU; the next step): every checkpoint scores the images again after a JPEG
  re-encode, a blur and a matched sharpen, and gives margin Grad-CAM maps.
* audit (manual): numbered overlays for a person to label where each map's hot spot sits.

    python -m src.shortcut                   # covariates and regions -> results/shortcut/
    python -m src.shortcut --audit-sheets    # the audit's numbered sheets and label template
    python -m src.shortcut --check           # self-check on synthetic data
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

from src.analysis import (SEEDS, direction_lines, experiment_ids, interval, load_runs, paired_mean, pairs,
                          subset_name, subset_weights, views, write_csv)
from src.augment import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_transform, resize_steps, tensor_steps
from src.config import AUGMENTATIONS, LOG_LABELS, MODELS, TRAIN_SPLITS
from src.data import LABELS, ChestXrayDataset
from src.evaluate import margins, results_directory
from src.figures import REGIME_COLOURS, REGIME_MARKERS, save_figure, titled
from src.gradcam import upsample
from src.splits import SEED, load_split
from src.train import run_directory

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


def region_arrays(runs, rows, root=None):
    """Return {(ID, seed): per-image region values} of every run's clean true-class maps.

    The maps are results/runs/<run>/stability_maps.npy[:, 0] - the 468 subset images, in the order
    of the run's stability records - and each image's background cells come from covariates.csv.
    """
    test = {row["image"]: row for row in rows if row["split"] == "test"}
    arrays = {}
    for key, run in runs.items():
        names = [subset_name(record) for record in run["records"]]
        # Each image's 49 background flags, stored as a string of 0s and 1s, back to a 7x7 grid.
        background = np.array([[cell == "1" for cell in test[name]["background"]] for name in names])
        maps = np.load(results_directory(root, "runs") / run_directory(run["config"]).name / "stability_maps.npy")[:, 0]
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
        "#15). The intervention (Colab) and the audit are added in later steps.",
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
              "out and counted.", "",
              "| Architecture | Class | Share on background cells | Share on top corners and bottom row | "
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
    lines += direction_lines(comparisons, groups, REGION_MEASURES)
    return "\n".join(lines) + "\n"


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
    """Compute the covariates and the regions and write results/shortcut/; return the summary text."""
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

    directory = results_directory(root, "shortcut")
    write_csv(table, directory / "availability.csv")
    write_csv(correlations, directory / "correlations.csv")
    write_csv(shifts, directory / "shift_correlations.csv")
    write_csv(per_run, directory / "regions.csv")
    write_csv(comparisons, directory / "region_comparisons.csv")
    text = summary(rows, table, correlations, shifts, per_run, comparisons, difference)
    (directory / "summary.md").write_text(text, encoding="utf-8")
    availability_figure(table, directory / "shortcut_availability.png")
    regions_figure(per_run, "background_share", "share of the map\non background cells",
                   "Where the true-class Grad-CAM maps sit: their share on the image's background cells, both seeds",
                   directory / "shortcut_regions_background.png")
    regions_figure(per_run, "edge_share", "share of the map on the top\ncorners and bottom row",
                   "Where the true-class Grad-CAM maps sit: their share on the top corners and bottom row, both seeds",
                   directory / "shortcut_regions_edge.png")
    print(f"  preprocessing against the evaluation transform: largest difference {difference:.1e}")
    print(f"{len(rows)} images, {len(correlations)} correlation rows, {len(per_run)} region rows written to {directory}")
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


def audit_sheets(root=None):
    """Write the audit's numbered overlay sheets, its key and a label template to results/shortcut/audit/.

    Every overlay pairs the image with its true-class map, scaled to the map's own maximum as in the
    Grad-CAM examples figure. Nothing on a sheet names the run or the class: which overlay is which
    is in key.csv, not opened while labelling (NOTES.md #15).
    """
    runs = load_runs(root)
    directory = results_directory(root, "shortcut") / "audit"
    directory.mkdir(exist_ok=True)
    names = [subset_name(record) for record in next(iter(runs.values()))["records"]]
    positions = audit_images(names)
    dataset = ChestXrayDataset(load_split("stability_subset"), unnormalised_transform())
    pictures = {position: dataset[position][0][0].numpy() for position in positions}
    items = [(key, position) for key in audit_runs() for position in positions]
    # A second seeded generator shuffles the order, so neighbouring overlays are not from one run.
    random.Random(f"{SEED}:audit order").shuffle(items)
    maps = {key: np.load(results_directory(root, "runs") / run_directory(runs[key]["config"]).name
                         / "stability_maps.npy")[:, 0] for key in audit_runs()}
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
            key, position = items[start + slot]
            cam = maps[key][position]
            axes[0].imshow(pictures[position], cmap="gray", vmin=0, vmax=1)
            axes[1].imshow(pictures[position], cmap="gray", vmin=0, vmax=1)
            heat = upsample(torch.from_numpy(cam[None]))[0].numpy()
            axes[1].imshow(heat / max(float(cam.max()), 1e-12), cmap="inferno", alpha=0.45, vmin=0, vmax=1)
            axes[0].set_title(f"#{number}", fontsize=9)
            axes[1].set_title("empty map" if cam.max() <= 0 else "", fontsize=7)
            key_rows.append({"number": number, "experiment": key[0], "seed": key[1], "image": names[position],
                             "map": "true class"})
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
        "second label is optional. Do not open `key.csv` while labelling: it says which run and image each\n"
        "overlay is.\n", encoding="utf-8")
    print(f"{len(items)} overlays on {len(range(0, len(items), AUDIT_PER_SHEET))} sheets written to {directory}")


def self_check():
    """Check the regions, the image features, the partial correlation and the audit draw on synthetic data."""
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


def main():
    """Run the local parts, write the audit sheets, or self-check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit-sheets", action="store_true", help="write the audit's overlay sheets and templates")
    parser.add_argument("--check", action="store_true", help="self-check on synthetic data")
    parser.add_argument("--output-root", type=Path, default=None, help="where results/ lives")
    args = parser.parse_args()
    if args.check:
        self_check()
    elif args.audit_sheets:
        audit_sheets(args.output_root)
    else:
        analyse(args.output_root)


if __name__ == "__main__":
    main()
