"""Explanation stability: how much a Grad-CAM heatmap moves under a small input perturbation.

Measured two ways on a fixed class-balanced test subset, and both are reported (NOTES.md, the
stability decision and question #7):

* In-family - a small rotation (+5 and -5 degrees, the same kernel as training's augmentation).
  The perturbed heatmap is rotated back into alignment and only pixels that stay inside the frame
  through both rotations are compared; otherwise even a perfectly consistent model would score low
  simply because the lungs moved.
* Out-of-family - small Gaussian noise, which no augmentation regime trains on and which moves no
  pixels, so the heatmaps compare directly.

Augmentation trains invariance to specific changes, so an in-family perturbation flatters augmented
models by construction; the gap between the two families is itself a finding. Every map explains
the image's true class, so all 27 configs answer the same question on the same image. Similarity is
the Pearson correlation of the two maps (chance level about 0); cosine is stored beside it.

    python -m src.stability                      # self-check (CPU, no checkpoint)
    python -m src.stability --write-subset       # draw the fixed test subset once (filenames only)
    python -m src.stability --calibrate --id T04 --id T13 [--output calibration.png]

--calibrate runs on validation images only and applies the rule in NOTES.md #7 for the noise level.
In an experiment, run_experiment.py calls evaluate_subset() after the test pass.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

from src.augment import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_transform
from src.config import get_config
from src.data import ChestXrayDataset, build_dataloader
from src.evaluate import BOOTSTRAP_SAMPLES, CONFIDENCE, classify, patient_groups
from src.figures import save_figure, titled
from src.gradcam import CLASS_NAMES, gradcam, upsample
from src.models import build_model
from src.splits import SEED, SPLIT_DIR, load_split
from src.train import get_device, load_best, run_directory

# The committed list of test images every experiment's stability is measured on.
SUBSET_NAME = "stability_subset"
PER_CLASS = 234
# In-family: mild augmentation's rotation bound, so the rotation is in-family for both augmented
# regimes. Both directions are measured, so no random sign is needed.
ANGLE = 5.0
PASSES = 2
# Out-of-family: the noise standard deviation in 0-1 pixel units. Fixed by the calibration rule in
# NOTES.md #7, on validation, before any model scores the test subset - so it starts unset, and an
# experiment refuses to run until the calibrated value is committed here.
NOISE_SIGMA = None
NOISE_SEED = SEED
CALIBRATION_SIGMAS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)
# At most this many of the 404 calibration images may change their predicted class (about 1%).
FLIP_LIMIT = 4


def draw_subset():
    """Return the stability subset as paths relative to the data root, NORMAL first.

    Every NORMAL test image and PER_CLASS of the PNEUMONIA ones, drawn with Python's random module
    seeded by text, as src/splits.py does - its stream does not change between NumPy versions.
    Only the committed test list is read; no image is opened.
    """
    lines = [line for line in (SPLIT_DIR / "test.txt").read_text(encoding="utf-8").splitlines() if line]
    # Path(line).parent.name is the class folder, e.g. "test/NORMAL/IM-0001.jpeg" -> "NORMAL".
    normal = sorted(line for line in lines if Path(line).parent.name == "NORMAL")
    pneumonia = sorted(line for line in lines if Path(line).parent.name == "PNEUMONIA")
    if len(normal) != PER_CLASS:
        raise ValueError(f"expected {PER_CLASS} NORMAL test images, found {len(normal)}")
    chosen = random.Random(f"{SEED}:stability:PNEUMONIA").sample(pneumonia, PER_CLASS)
    return normal + sorted(chosen)


def write_subset():
    """Write data/splits/stability_subset.txt from draw_subset() and return its path."""
    path = SPLIT_DIR / f"{SUBSET_NAME}.txt"
    # newline="\n" keeps the committed file byte-identical on Windows and Linux.
    path.write_text("\n".join(draw_subset()) + "\n", encoding="utf-8", newline="\n")
    return path


def unit_scale(images):
    """Return normalised images mapped back to 0-1 pixel values (the ImageNet normalisation undone)."""
    # view(1, 3, 1, 1) lets one mean and std per channel broadcast over a whole batch.
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return images * std + mean


def normalise(units):
    """Return 0-1 pixel images normalised the way the models were trained on."""
    mean = torch.tensor(IMAGENET_MEAN, device=units.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=units.device).view(1, 3, 1, 1)
    return (units - mean) / std


def rotate(images, angle):
    """Rotate a batch about its centre by angle degrees (counter-clockwise), bilinear, black fill.

    The same interpolation, centre and fill as training's RandomAffine; training rotates the
    8-bit image and this the float tensor, which differ only at quantisation level.
    """
    return TF.rotate(images, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)


def rotation_mask(angle, size=IMAGE_SIZE):
    """Return a (size, size) bool mask of the pixels that stay inside the frame when rotated and back.

    Rotating a frame of ones and rotating it back leaves exactly 1 only where no black fill or
    blending with the fill reached: the corners and a thin edge ring drop out.
    """
    ones = torch.ones(1, 1, size, size)
    return rotate(rotate(ones, angle), -angle)[0, 0] > 1 - 1e-6


def noise(position, draw, sigma, size=IMAGE_SIZE):
    """Return one (1, size, size) Gaussian noise image for an image position and draw.

    Drawn on the CPU from a generator seeded by (NOISE_SEED, position, draw), so every config and
    every device sees exactly the same noise. One draw serves all three channels, because the
    images are grayscale copied three times.
    """
    generator = torch.Generator().manual_seed(NOISE_SEED * 1_000_000 + position * 10 + draw)
    return torch.randn(1, size, size, generator=generator) * sigma


def perturbations(units, positions, angle, sigma, families=("in", "out")):
    """Return [(family, perturbed 0-1 images, angle to undo)] for a batch's perturbed passes.

    Two rotations (+angle, -angle) form the in-family passes, two noise draws the out-of-family
    ones. Noise is added in 0-1 pixel space and clipped back into it.
    """
    passes = []
    if "in" in families:
        passes += [("in", rotate(units, sign * angle), -sign * angle) for sign in (1, -1)]
    if "out" in families:
        for draw in range(PASSES):
            # expand(3, ...) repeats the one noise channel for R, G and B without copying memory.
            added = torch.stack([noise(position, draw, sigma, units.shape[-1]).expand(3, -1, -1)
                                 for position in positions])
            passes.append(("out", (units + added.to(units.device)).clamp(0, 1), 0.0))
    return passes


def similarity(first, second, mask=None):
    """Return (Pearson, cosine) of two maps over the mask's pixels, in float64; NaN where undefined.

    Pearson compares the maps' patterns around their own means, so two unrelated maps score about
    0; cosine does not subtract the mean, and two non-negative maps score well above 0 even when
    unrelated. Both are undefined for a constant map, which includes an empty one.
    """
    first, second = first.double(), second.double()
    if mask is not None:
        first, second = first[mask], second[mask]
    first, second = first.flatten(), second.flatten()
    norms = first.norm() * second.norm()
    cosine = float(first @ second / norms) if norms > 0 else float("nan")
    first, second = first - first.mean(), second - second.mean()
    norms = first.norm() * second.norm()
    pearson = float(first @ second / norms) if norms > 0 else float("nan")
    return pearson, cosine


def predict_classes(model, images, threshold):
    """Return the predicted class of each image at a logit-margin threshold, as a CPU tensor."""
    with torch.no_grad():
        return torch.from_numpy(classify(model(images).cpu(), threshold))


def measure_batch(model, name, images, labels, positions, threshold, angle, sigma, families=("in", "out")):
    """Return (per-image records, raw maps (N, 5, 7, 7)) for one batch of normalised images.

    The maps explain each image's true class: clean, then the four perturbed passes. A record holds
    the clean prediction, and for every pass its Pearson, cosine, prediction flip and mean absolute
    pixel change (in 0-1 units, a model-free measure of how big the perturbation was).
    """
    labels = labels.to(images.device)
    clean_maps, _ = gradcam(model, name, images, labels)
    clean_predicted = predict_classes(model, images, threshold)
    clean_large = upsample(clean_maps).cpu()
    units = unit_scale(images)
    raw = [clean_maps.cpu()]
    records = [{"class": CLASS_NAMES[int(label)], "predicted": CLASS_NAMES[int(guess)],
                "correct": bool(guess == label.cpu()), "passes": []}
               for label, guess in zip(labels, clean_predicted)]
    for family, perturbed, undo in perturbations(units, positions, angle, sigma, families):
        inputs = normalise(perturbed)
        maps, _ = gradcam(model, name, inputs, labels)
        predicted = predict_classes(model, inputs, threshold)
        raw.append(maps.cpu())
        large = upsample(maps).cpu()
        mask = None
        if family == "in":
            # Undo the rotation on the heatmap, then compare only where both frames overlap.
            large = rotate(large.unsqueeze(1), undo).squeeze(1)
            mask = rotation_mask(-undo)
        change = (perturbed - units).abs().flatten(1).mean(dim=1).cpu()
        for index, record in enumerate(records):
            pearson, cosine = similarity(clean_large[index], large[index], mask)
            record["passes"].append({"family": family, "pearson": pearson, "cosine": cosine,
                                     "flipped": bool(predicted[index] != clean_predicted[index]),
                                     "pixel_change": float(change[index])})
    # stack gives (5, N, 7, 7); transpose(0, 1) puts the image first: (N, 5, 7, 7).
    return records, torch.stack(raw).transpose(0, 1)


def family_score(record, family, measure="pearson"):
    """Return the mean of an image's defined passes in one family, or NaN if none is defined."""
    values = [item[measure] for item in record["passes"] if item["family"] == family]
    values = [value for value in values if not np.isnan(value)]
    return float(np.mean(values)) if values else float("nan")


def bootstrap_mean(values, groups, samples, seed):
    """Return the (low, high) 95% percentile interval of the mean of values over patient groups.

    Resamples the patient groups with replacement, as src/evaluate.py does for every other
    interval; NaN values (images outside the common set) are ignored inside each resample.
    """
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        drawn = rng.integers(0, len(groups), len(groups))
        chosen = values[np.concatenate([groups[position] for position in drawn])]
        chosen = chosen[~np.isnan(chosen)]
        if len(chosen):
            means.append(chosen.mean())
    tail = (100 - CONFIDENCE) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def summarise(records, paths, clean_maps, sigma, angle, samples=BOOTSTRAP_SAMPLES, seed=SEED):
    """Return the stability fields for a results file, from every image's record.

    The headline means are over the common set - images with at least one defined pass in both
    families - so in-family, out-of-family and their gap describe the same images. clean_maps are
    the (N, 7, 7) clean maps, for the chance baseline.
    """
    inside = np.array([family_score(record, "in") for record in records])
    outside = np.array([family_score(record, "out") for record in records])
    common = ~np.isnan(inside) & ~np.isnan(outside)
    gap = np.where(common, inside - outside, np.nan)
    groups = patient_groups(paths)
    fields = {}
    for key, values in (("in_family", inside), ("out_of_family", outside), ("gap", gap)):
        values = np.where(common, values, np.nan)
        low, high = bootstrap_mean(values, groups, samples, seed)
        name = f"gradcam_stability_{key}"
        fields.update({name: float(np.nanmean(values)), f"{name}_ci_low": low, f"{name}_ci_high": high})

    correct = np.array([record["correct"] for record in records])
    both = common & correct
    passes = [item for record in records for item in record["passes"]]

    def rate(family, key):
        """Return the mean of one per-pass quantity over a family's passes."""
        return float(np.mean([item[key] for item in passes if item["family"] == family]))

    # Chance baseline: each clean map against the next same-class image's clean map, so two
    # unrelated images show what "similar" means for these maps before any perturbation.
    large = upsample(clean_maps)
    baseline = []
    for index, record in enumerate(records):
        same = [other for other, item in enumerate(records) if item["class"] == record["class"]]
        partner = same[(same.index(index) + 1) % len(same)]
        baseline.append(similarity(large[index], large[partner]))
    baseline = np.array(baseline)

    fields["stability"] = {
        "similarity": "pearson", "target": "true class", "subset": f"data/splits/{SUBSET_NAME}.txt",
        "images": len(records), "rotation_degrees": angle, "noise_sigma": sigma, "passes_per_family": PASSES,
        "noise_seed": NOISE_SEED, "common_images": int(common.sum()),
        "excluded_in_family": int(np.isnan(inside).sum()), "excluded_out_of_family": int(np.isnan(outside).sum()),
        "correct_clean": int(correct.sum()),
        "correct_only": {"images": int(both.sum()),
                         "in_family": float(inside[both].mean()) if both.any() else None,
                         "out_of_family": float(outside[both].mean()) if both.any() else None,
                         "gap": float(gap[both].mean()) if both.any() else None},
        "cosine": {family: float(np.nanmean(np.where(common, [family_score(record, key, "cosine")
                                                              for record in records], np.nan)))
                   for family, key in (("in_family", "in"), ("out_of_family", "out"))},
        "flip_rate": {"in_family": rate("in", "flipped"), "out_of_family": rate("out", "flipped")},
        "mean_pixel_change": {"in_family": rate("in", "pixel_change"), "out_of_family": rate("out", "pixel_change")},
        "chance_baseline": {"pearson": float(np.nanmean(baseline[:, 0])), "cosine": float(np.nanmean(baseline[:, 1]))},
    }
    return fields


def evaluate_subset(model, config, threshold, run_dir, workers=0, batch_size=32, sigma=None, angle=ANGLE):
    """Measure stability on the fixed test subset and save the per-image record and raw maps.

    Returns (fields for the results file, examples) - examples holds the first four images of each
    class with their clean maps and predictions, for the Grad-CAM examples figure. Refuses to run
    before the noise level has been calibrated (NOTES.md #7).
    """
    sigma = NOISE_SIGMA if sigma is None else sigma
    if sigma is None:
        raise RuntimeError("NOISE_SIGMA is not set: run the calibration in NOTES.md #7 and commit its value first")
    device = next(model.parameters()).device
    loader = build_dataloader(SUBSET_NAME, batch_size, build_transform(config.augmentation, train=False),
                              num_workers=workers)
    records, raw, start = [], [], 0
    for images, labels in loader:
        positions = list(range(start, start + len(labels)))
        batch_records, batch_maps = measure_batch(model, config.model, images.to(device), labels,
                                                  positions, threshold, angle, sigma)
        records.extend(batch_records)
        raw.append(batch_maps)
        start += len(labels)
    raw = torch.cat(raw)
    paths = loader.dataset.paths
    for record, path in zip(records, paths):
        record["image"] = path.name

    # Written before the results file, so a finished results file always has both beside it.
    (run_dir / "stability_per_image.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    torch.save(raw, run_dir / "stability_maps.pt")

    labels = np.array([record["class"] for record in records])
    chosen = [int(index) for name in ("NORMAL", "PNEUMONIA") for index in np.flatnonzero(labels == name)[:4]]
    examples = {"images": torch.stack([loader.dataset[index][0] for index in chosen]),
                "maps": raw[chosen, 0],
                "true": torch.tensor([int(labels[index] == "PNEUMONIA") for index in chosen]),
                "predicted": torch.tensor([int(records[index]["predicted"] == "PNEUMONIA") for index in chosen])}
    return summarise(records, paths, raw[:, 0], sigma, angle, seed=config.seed), examples


def calibration_sample():
    """Return the validation images used for calibration: every NORMAL one and as many PNEUMONIA."""
    paths = load_split("val")
    normal = [path for path in paths if path.parent.name == "NORMAL"]
    pneumonia = [path for path in paths if path.parent.name == "PNEUMONIA"]
    return normal + sorted(random.Random(f"{SEED}:stability:calibration").sample(pneumonia, len(normal)))


def calibrate(run_ids, output=None):
    """Print the calibration table for NOTES.md #7 and return the noise level its rule selects.

    Validation images only. Model-free: the mean absolute pixel change of the rotation and of each
    candidate sigma. Per checkpoint: how many images change predicted class (at margin 0) under
    the rotation and each sigma, and the mean Pearson per family.
    """
    device, device_name = get_device()
    paths = calibration_sample()
    dataset = ChestXrayDataset(paths, build_transform("none", train=False))
    images = torch.stack([dataset[index][0] for index in range(len(dataset))])
    labels = torch.tensor(dataset.labels)
    positions = list(range(len(paths)))
    units = unit_scale(images)
    rotation_change = np.mean([(rotate(units, sign * ANGLE) - units).abs().mean().item() for sign in (1, -1)])
    sigma_change = {sigma: np.mean([(noisy - units).abs().mean().item() for _, noisy, _ in
                                    perturbations(units, positions, ANGLE, sigma, families=("out",))])
                    for sigma in CALIBRATION_SIGMAS}
    print(f"calibration on {len(paths)} validation images ({device_name}); mean absolute pixel change: "
          f"rotation +/-{ANGLE:g} deg {rotation_change:.4f}")
    for sigma, change in sigma_change.items():
        print(f"  noise sigma {sigma:.2f}: {change:.4f}")

    flips = {}
    for run_id in run_ids:
        config = get_config(run_id)
        model = load_best(config, run_directory(config), device).eval()
        rows = []
        for sigma in (None,) + CALIBRATION_SIGMAS:
            # The first row is the rotation, the rest one row per candidate sigma.
            family = "in" if sigma is None else "out"
            records = []
            for start in range(0, len(paths), 64):
                batch = slice(start, start + 64)
                batch_records, _ = measure_batch(model, config.model, images[batch].to(device), labels[batch],
                                                 positions[batch], 0.0, ANGLE, sigma or 0.0, families=(family,))
                records.extend(batch_records)
            changed = sum(any(item["flipped"] for item in record["passes"] if item["family"] == family)
                          for record in records)
            score = np.nanmean([family_score(record, family) for record in records])
            rows.append((sigma, changed, score))
        flips[run_id] = {sigma: changed for sigma, changed, _ in rows}
        print(f"{run_id} ({config.model}): images whose class changes, of {len(paths)}; mean Pearson")
        for sigma, changed, score in rows:
            label = f"rotation +/-{ANGLE:g} deg" if sigma is None else f"noise sigma {sigma:.2f}"
            print(f"  {label:22s} {changed:3d}   {score:+.3f}")

    rotation_ok = all(flips[run_id][None] <= FLIP_LIMIT for run_id in run_ids)
    allowed = [sigma for sigma in CALIBRATION_SIGMAS if all(flips[run_id][sigma] <= FLIP_LIMIT for run_id in run_ids)]
    closest = min(CALIBRATION_SIGMAS, key=lambda sigma: abs(sigma_change[sigma] - rotation_change))
    if not rotation_ok:
        print(f"RULE: the rotation itself changes more than {FLIP_LIMIT} predictions on a checkpoint - stop and discuss")
        chosen = None
    elif closest in allowed:
        chosen = closest
        print(f"RULE: sigma {chosen:.2f} - its pixel change is closest to the rotation's, and it changes at most "
              f"{FLIP_LIMIT} predictions on every checkpoint")
    elif allowed:
        chosen = max(allowed)
        print(f"RULE: the closest match ({closest:.2f}) changes too many predictions; sigma {chosen:.2f} is the "
              f"largest within the limit")
    else:
        print("RULE: every sigma changes too many predictions - stop and discuss")
        chosen = None
    if output is not None and chosen is not None:
        calibration_figure(units, labels, chosen, output)
    return chosen


def calibration_figure(units, labels, sigma, output):
    """Draw one NORMAL and one PNEUMONIA validation image clean, rotated both ways and with noise."""
    picks = [int(np.flatnonzero(labels.numpy() == index)[0]) for index in sorted(CLASS_NAMES)]
    columns = [("clean", lambda batch, pos: batch)]
    columns += [(f"rotated {sign * ANGLE:+g} deg", lambda batch, pos, s=sign: rotate(batch, s * ANGLE)) for sign in (1, -1)]
    columns += [(f"noise sigma {sigma:.2f}", lambda batch, pos: perturbations(batch, pos, ANGLE, sigma, ("out",))[0][1])]
    figure = Figure(figsize=(2.4 * len(columns), 5.0), layout="constrained")
    grid = figure.subplots(len(picks), len(columns), squeeze=False)
    for row, pick in enumerate(picks):
        for column, (label, apply) in enumerate(columns):
            picture = apply(units[pick:pick + 1], [pick])[0, 0].numpy()
            grid[row, column].imshow(picture, cmap="gray", vmin=0, vmax=1)
            grid[row, column].set_title(f"{CLASS_NAMES[int(labels[pick])]}, {label}", fontsize=7)
            grid[row, column].set_xticks([])
            grid[row, column].set_yticks([])
    title = f"Stability perturbations (validation images): rotation +/-{ANGLE:g} deg, noise sigma {sigma:.2f}"
    titled(figure, title, size=8)
    save_figure(figure, output, title)


def self_check():
    """Check the measurement on synthetic maps and a real architecture, without any checkpoint."""
    fresh, committed = draw_subset(), (SPLIT_DIR / f"{SUBSET_NAME}.txt")
    if committed.exists():
        lines = [line for line in committed.read_text(encoding="utf-8").splitlines() if line]
        test = set((SPLIT_DIR / "test.txt").read_text(encoding="utf-8").split())
        normal_test = {line for line in test if Path(line).parent.name == "NORMAL"}
        print(f"  subset file equals a fresh draw: {lines == fresh}; {len(set(lines))} distinct test images "
              f"({sum(Path(line).parent.name == 'NORMAL' for line in lines)} NORMAL, all of them: "
              f"{normal_test <= set(lines)}; all from test.txt: {set(lines) <= test})")
    else:
        print("  subset file not written yet (python -m src.stability --write-subset)")

    torch.manual_seed(0)
    blob = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
    blob[60:120, 80:150] = 1.0
    blob = upsample(torch.nn.functional.avg_pool2d(blob[None, None], 32)[0])[0]
    mask = rotation_mask(ANGLE)
    back = rotate(rotate(blob[None, None], ANGLE), -ANGLE)[0, 0]
    print(f"  map rotated +{ANGLE:g} and back, Pearson on the mask: {similarity(blob, back, mask)[0]:.6f} (> 0.999: "
          f"{similarity(blob, back, mask)[0] > 0.999}); mask drops the corners: {not mask[0, 0] and bool(mask[112, 112])}")
    first, second = torch.rand(7, 7), torch.rand(7, 7)
    pearson, cosine = similarity(upsample(first[None])[0], upsample(second[None])[0])
    print(f"  unrelated random maps: Pearson {pearson:+.3f} (near 0), cosine {cosine:.3f} (well above 0)")
    print(f"  an empty map is undefined: {np.isnan(similarity(torch.zeros(7, 7), first)[0])}")
    one, two = noise(3, 1, 0.05), noise(3, 1, 0.05)
    # Grayscale test images - one channel copied three times, as every X-ray reaches the model.
    images = torch.rand(2, 1, 16, 16).expand(-1, 3, -1, -1)
    noisy = perturbations(images, [0, 1], ANGLE, 0.05, ("out",))[0][1]
    print(f"  noise identical on repeated calls: {torch.equal(one, two)}; the three channels equal: "
          f"{torch.equal(noisy[:, 0], noisy[:, 1]) and torch.equal(noisy[:, 1], noisy[:, 2])}")

    model = build_model("resnet18").eval()
    # Pixel values inside 0-1, so noise of sigma 0 plus clipping leaves the input unchanged.
    images = normalise(torch.rand(4, 3, IMAGE_SIZE, IMAGE_SIZE))
    labels = torch.tensor([0, 1, 0, 1])
    records, maps = measure_batch(model, "resnet18", images, labels, [0, 1, 2, 3], 0.0, 0.0, 0.0)
    scores = [item["pearson"] for record in records for item in record["passes"]]
    # Empty maps (common for an untrained head on noise images) are undefined, not 1.
    identical = [value for value in scores if not np.isnan(value)]
    # Undoing and redoing the normalisation moves the input by about 2e-7 in float32, which costs
    # about 2e-12 of Pearson - hence 1e-9, not exact equality.
    print(f"  identical input (angle 0, sigma 0) scores 1: "
          f"{bool(identical) and all(abs(value - 1) < 1e-9 for value in identical)} "
          f"({len(identical)} defined, {len(scores) - len(identical)} undefined from empty maps); "
          f"raw maps {tuple(maps.shape)}")
    records, maps = measure_batch(model, "resnet18", images, labels, [0, 1, 2, 3], 0.0, ANGLE, 0.05)
    fields = summarise(records, [Path(f"test/NORMAL/IM-{i:04d}-0001.jpeg") for i in range(4)], maps[:, 0],
                       0.05, ANGLE, samples=200)
    inside = np.array([family_score(record, "in") for record in records])
    outside = np.array([family_score(record, "out") for record in records])
    print(f"  gap equals in minus out on the common set: "
          f"{abs(fields['gradcam_stability_gap'] - (np.nanmean(inside) - np.nanmean(outside))) < 1e-12}; "
          f"intervals contain the means: "
          f"{all(fields[f'{k}_ci_low'] - 1e-12 <= fields[k] <= fields[f'{k}_ci_high'] + 1e-12 for k in ('gradcam_stability_in_family', 'gradcam_stability_out_of_family'))}")


def main():
    """Self-check, write the subset, or run the noise calibration, depending on the arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-subset", action="store_true", help="draw and write data/splits/stability_subset.txt")
    parser.add_argument("--calibrate", action="store_true", help="run the noise calibration on validation images")
    parser.add_argument("--id", action="append", default=[], help="a tuning run to calibrate on (repeat)")
    parser.add_argument("--output", type=Path, default=None, help="where to write the calibration figure")
    args = parser.parse_args()
    if args.write_subset:
        print(f"wrote {write_subset()}")
    elif args.calibrate:
        if not args.id:
            parser.error("--calibrate needs at least one --id")
        calibrate(args.id, args.output)
    else:
        self_check()


if __name__ == "__main__":
    main()
