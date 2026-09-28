"""Turn the two sweeps into the report's evidence: collect per-image scores, then compare runs.

The results files hold each run's metrics but not its per-image scores, and a difference between
two runs needs both runs scored on the same resampled images. So this module works in two stages:

* collect (Colab, GPU): every experiment's best checkpoint scores validation and test again - the
  exact evaluation path - and the result is checked against the stored results file before it is
  kept. Each run's small files are gathered beside the scores under results/runs/<ID>_seed<N>/, so
  the archive in the repo holds everything the comparisons read.
* analyse (locally, CPU, from the archive): every comparison the report draws on - augmentation
  regime, data size, threshold, seed, architecture (levels, and how the larger models respond
  differently), and the regime effect by data size - as a paired difference between two runs, with a
  95% patient-group bootstrap
  interval, for each seed and for the two seeds together (NOTES.md #32). It writes
  results/analysis/: comparisons.csv, runs.csv, summary.md and six figures.

    python -m src.analysis --collect --seed 42 --workers 2    # Colab; add --id E01 for one run
    python -m src.analysis                                     # the analysis (CPU, ~20 min the first time)
    python -m src.analysis --check                             # self-check on synthetic data
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torchvision
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.config import (AUGMENTATIONS, EXPERIMENTS, LOG_LABELS, MODELS, RECIPE, TRAIN_SPLITS, get_config,
                        output_stem)
from src.evaluate import (BOOTSTRAP_SAMPLES, CONFIDENCE, choose_threshold, classify, confusion_counts, margins,
                          patient_groups, results_directory, score_split, scores)
from src.figures import REGIME_COLOURS, REGIME_MARKERS, save_figure, titled
from src.paths import code_version
from src.splits import SEED
from src.stability import family_score
from src.train import get_device, load_best, run_directory

# Copied unchanged from a run's checkpoint folder. The raw maps are converted instead, because
# .gitignore keeps every *.pt file out of the repo.
RUN_FILES = ("stability_per_image.json", "validation_metrics.json", "history.json")
# The re-scoring gate: equal to float rounding, or equal up to GPU or library drift.
EXACT, APPROX = 1e-12, 1e-6

# The two sweeps (NOTES.md #21, #32).
SEEDS = (SEED, 43)
# Every classification metric, in the order evaluate.scores() returns them: the seven at 0.5, then
# the six threshold metrics at each run's validation-chosen threshold (ROC-AUC has no threshold).
METRICS = ("roc_auc", "balanced_accuracy", "accuracy", "precision", "recall", "specificity", "f1")
CLASSIFICATION = METRICS + tuple(f"{name}_tuned" for name in METRICS[1:])
STABILITY = ("in_family", "out_of_family", "gap")
# Which images a stability difference averages over: those defined in both runs ("all"), only those
# both runs also classify correctly at 0.5 ("correct"), or all of them with an undefined pass counted
# as 0 ("empty_zero") - NOTES.md, how the stability results are read.
VARIANTS = ("all", "correct", "empty_zero")
# (second, first): every difference is second minus first.
REGIME_CONTRASTS = (("mild", "none"), ("strong", "none"), ("strong", "mild"))
SIZE_CONTRASTS = (("train_50", "train_20"), ("train_100", "train_20"))
# The measures summary.md tabulates: (metric, variant, how the table names it).
MEASURES = (
    ("roc_auc", "", "ROC-AUC"),
    ("balanced_accuracy", "", "balanced accuracy @0.5"),
    ("balanced_accuracy_tuned", "", "balanced accuracy, tuned threshold"),
    ("specificity", "", "specificity @0.5"),
    ("recall", "", "recall @0.5"),
    ("stability_in_family", "all", "stability in-family"),
    ("stability_out_of_family", "all", "stability out-of-family"),
    ("stability_gap", "all", "stability gap"),
    ("stability_gap", "correct", "gap, images correct in both runs"),
    ("stability_gap", "empty_zero", "gap, empty map = 0"),
    ("stability_out_of_family", "empty_zero", "out-of-family, empty map = 0"),
    ("empty_maps_out", "all", "share of empty maps under noise"),
)
# Larger minus smaller, by parameter count: MobileNetV3-Small 2.5M < EfficientNet-B0 5.3M < ResNet18 11.7M.
ARCHITECTURE_CONTRASTS = (("efficientnet_b0", "mobilenet_v3_small"), ("resnet18", "mobilenet_v3_small"),
                          ("resnet18", "efficientnet_b0"))
# The shorter measure lists of the follow-up tables (NOTES.md #33 and the proposal's third question).
KEY_MEASURES = (
    ("roc_auc", "", "ROC-AUC"),
    ("balanced_accuracy_tuned", "", "balanced accuracy, tuned threshold"),
    ("specificity", "", "specificity @0.5"),
    ("stability_gap", "all", "stability gap"),
)
# Two architectures start from different stability gaps (the reading rule), so levels are classification only.
LEVEL_MEASURES = KEY_MEASURES[:3] + (("recall", "", "recall @0.5"),)
SHIFT_MEASURES = (
    ("specificity", "", "test specificity @0.5"),
    ("specificity_tuned", "", "test specificity, tuned threshold"),
    ("recall", "", "test recall @0.5"),
    ("roc_auc", "", "test ROC-AUC"),
    ("validation_specificity", "", "validation specificity @0.5"),
    ("validation_roc_auc", "", "validation ROC-AUC"),
    ("normal_margin_median_val", "", "NORMAL images' median margin, validation"),
    ("normal_margin_median_test", "", "NORMAL images' median margin, test"),
)
# NOTES.md #35 keeps ranking, operating point and stability apart. Tuned balanced accuracy is left out: its
# threshold does not carry over to test (#31, #33), so it would mix the first two.
SIZE_MEASURES = (
    ("roc_auc", "", "ROC-AUC"),
    ("specificity", "", "specificity @0.5"),
    ("normal_margin_median_test", "", "NORMAL images' median margin, test"),
    ("stability_gap", "all", "stability gap"),
    ("stability_gap", "empty_zero", "gap, empty map = 0"),
)


def image_names(paths):
    """Return "CLASS/filename" for each image path - enough to rebuild the patient groups later."""
    return [f"{path.parent.name}/{path.name}" for path in paths]


def split_record(logits, labels, paths):
    """Return one split's per-image scores as lists that json can write."""
    # .tolist() turns a tensor into nested Python lists; float32 values survive the trip exactly.
    return {"images": image_names(paths), "labels": labels.tolist(), "logits": logits.tolist()}


def gate(stored, val_logits, val_labels, logits, labels):
    """Return (verdict, largest difference): do re-scored logits reproduce a stored results file?

    The original logits were never saved, so the stored metrics are the only reference. The
    threshold is chosen again from the re-scored validation logits, and every test metric is
    recomputed at 0.5 and at the stored threshold. "PASS-exact" means every number matches to float
    rounding; "PASS-approx" allows drift from another GPU or library version that moves no
    prediction; anything else is "FAIL".
    """
    threshold = stored["threshold_logit_margin"]
    fresh = scores(logits, labels)
    fresh.update({f"{name}_tuned": value
                  for name, value in scores(logits, labels, threshold, with_auc=False).items()})
    differences = [abs(fresh[name] - stored[name]) for name in fresh]
    differences.append(abs(choose_threshold(val_logits, val_labels) - threshold))
    same_counts = (confusion_counts(logits, labels) == stored["confusion_matrix"]
                   and confusion_counts(logits, labels, threshold) == stored["confusion_matrix_tuned"])
    largest = max(differences)
    if same_counts and largest <= EXACT:
        return "PASS-exact", largest
    if same_counts and largest <= APPROX:
        return "PASS-approx", largest
    return "FAIL", largest


def collect(seed=SEED, root=None, workers=0, run_ids=None):
    """Re-score every experiment of one seed and gather its files; return {run ID: verdict}.

    A run whose scores.json already exists is skipped, so an interrupted session picks up where it
    stopped. Only results/runs/ is written; checkpoints and results files are only read.
    """
    # get_device also switches cuDNN to deterministic mode - before any model runs, exactly as the
    # evaluation that wrote the results files did.
    device, device_name = get_device()
    runs = results_directory(root, "runs")
    verdicts = {}
    for run_id in run_ids or EXPERIMENTS:
        config = get_config(run_id, seed)
        run_dir = run_directory(config, root)
        target = runs / run_dir.name
        if (target / "scores.json").exists():
            print(f"{run_dir.name}: already collected, skipping")
            continue
        metrics_path = results_directory(root) / f"{output_stem(config)}_metrics.json"
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
        model = load_best(config, run_dir, device)
        val_logits, val_labels, val_paths = score_split(model, "val", config, workers, RECIPE, device)
        logits, labels, paths = score_split(model, "test", config, workers, RECIPE, device)
        verdict, largest = gate(stored, val_logits, val_labels, logits, labels)

        target.mkdir(parents=True, exist_ok=True)
        for name in RUN_FILES:
            # copy2 copies the file and keeps its timestamps.
            shutil.copy2(run_dir / name, target / name)
        # The saved (468, 5, 7, 7) tensor; .numpy() hands the same numbers to NumPy, which saves them.
        maps = torch.load(run_dir / "stability_maps.pt", map_location="cpu")
        np.save(target / "stability_maps.npy", maps.numpy().astype(np.float32))
        payload = {
            "experiment_id": run_id, "seed": seed, "gate": verdict, "largest_difference": largest,
            "code_version": code_version(), "device": device_name, "torch": torch.__version__,
            "torchvision": torchvision.__version__, "stored_torch": stored["evaluation"]["torch"],
            "val": split_record(val_logits, val_labels, val_paths),
            "test": split_record(logits, labels, paths),
        }
        # Written last and by rename: scores.json marks a run as collected, so it must never exist
        # half written.
        temporary = target / "scores.json.tmp"
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(target / "scores.json")
        verdicts[run_id] = verdict
        print(f"{run_dir.name}: {verdict} (largest difference {largest:.1e}; torch {torch.__version__} "
              f"on {device_name}, the results file's torch {stored['evaluation']['torch']})")
    return verdicts


def subset_name(record):
    """Return a stability record's image as "CLASS/filename", the form scores.json uses."""
    return f"{record['class']}/{record['image']}"


def load_runs(root=None):
    """Return {(experiment ID, seed): run} for both sweeps, read from results/.

    A run is a dict: its config and results file, its test and validation logits and labels as
    tensors with the image names, its per-image stability records, its archived validation results
    file and per-epoch history, and its operating point (operating_point). Raises ValueError for a
    run that failed the re-scoring check, and for two runs that did not score the same images in
    the same order - pairing image by image is what every comparison rests on.
    """
    runs = {}
    for seed in SEEDS:
        for run_id in EXPERIMENTS:
            config = get_config(run_id, seed)
            folder = results_directory(root, "runs") / run_directory(config).name
            collected = json.loads((folder / "scores.json").read_text(encoding="utf-8"))
            if collected["gate"] == "FAIL":
                raise ValueError(f"{folder.name} failed the re-scoring check, so its scores cannot be used")
            metrics_path = results_directory(root) / f"{output_stem(config)}_metrics.json"
            runs[run_id, seed] = {
                "id": run_id, "seed": seed, "config": config,
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
                # float32, as the model produced them: JSON kept every float32 value exactly.
                "logits": torch.tensor(collected["test"]["logits"], dtype=torch.float32),
                "labels": torch.tensor(collected["test"]["labels"]),
                "images": collected["test"]["images"],
                "val_logits": torch.tensor(collected["val"]["logits"], dtype=torch.float32),
                "val_labels": torch.tensor(collected["val"]["labels"]),
                "val_images": collected["val"]["images"],
                "records": json.loads((folder / "stability_per_image.json").read_text(encoding="utf-8")),
                "stored_validation": json.loads((folder / "validation_metrics.json").read_text(encoding="utf-8")),
                "history": json.loads((folder / "history.json").read_text(encoding="utf-8"))["epochs"],
            }
            runs[run_id, seed].update(operating_point(runs[run_id, seed]))
    first = next(iter(runs.values()))
    subset = [subset_name(record) for record in first["records"]]
    for key, run in runs.items():
        # torch.equal is True only for tensors of the same shape and identical values.
        if (run["images"] != first["images"] or not torch.equal(run["labels"], first["labels"])
                or run["val_images"] != first["val_images"] or not torch.equal(run["val_labels"], first["val_labels"])
                or [subset_name(record) for record in run["records"]] != subset):
            raise ValueError(f"{key} scored different images from {first['id']} seed {first['seed']}")
    return runs


def operating_point(run):
    """Return where a run's decisions sit: its validation metrics at 0.5, and its NORMAL images' margins.

    The margin is the PNEUMONIA logit minus the NORMAL logit, and a model says PNEUMONIA above 0 (the
    0.5 threshold) or above its tuned threshold. The median and 90th percentile of the NORMAL images'
    margins show, without any threshold, how close they sit to being called pneumonia - on validation
    and on test (NOTES.md #33). Validation specificity cannot show it: it is 0.955-1.0 in every run.
    """
    values = {"validation": scores(run["val_logits"], run["val_labels"])}
    for split, logits, labels in (("val", run["val_logits"], run["val_labels"]), ("test", run["logits"], run["labels"])):
        # margins() gives every image's margin in float64; the label mask keeps the NORMAL ones.
        normal = margins(logits)[labels.numpy() == 0]
        values[f"normal_margin_median_{split}"] = float(np.median(normal))
        values[f"normal_margin_p90_{split}"] = float(np.percentile(normal, 90))
    return values


def test_draws(images, labels, samples=BOOTSTRAP_SAMPLES, seed=SEED):
    """Return the bootstrap resamples of the test set, each an array of image positions.

    Drawn exactly as evaluate.bootstrap_intervals draws them - the same generator, seed and patient
    groups, and a draw holding one class only skipped - so for the seed-42 runs, whose results files
    used seed 42, the intervals computed from these draws reproduce the stored ones exactly.
    """
    # patient_groups reads the class from the parent folder name, which "CLASS/filename" provides.
    groups = patient_groups([Path(name) for name in images])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        drawn = rng.integers(0, len(groups), len(groups))
        index = np.concatenate([groups[position] for position in drawn])
        # A draw of one class has no ROC-AUC; evaluate.py skips it, so this does too.
        if len(torch.unique(labels[index])) == 2:
            draws.append(index)
    return draws


def bootstrap_rows(run, draws):
    """Return a (draws, 13) array: every classification metric of one run on every resample.

    The columns follow CLASSIFICATION, and every value comes from evaluate.scores() - the function
    behind every number in the results files - at 0.5 and at the run's own tuned threshold.
    """
    threshold = run["metrics"]["threshold_logit_margin"]
    rows = []
    for index in draws:
        logits, labels = run["logits"][index], run["labels"][index]
        values = scores(logits, labels)
        values.update({f"{name}_tuned": value
                       for name, value in scores(logits, labels, threshold, with_auc=False).items()})
        rows.append([values[name] for name in CLASSIFICATION])
    return np.array(rows)


def bootstrap_cache(runs, draws, root=None):
    """Return {(ID, seed): bootstrap rows} for every run, computed once and then read from disk.

    All 54 runs take about 20 minutes on a laptop CPU, so the rows are kept in
    results/analysis/bootstrap_cache.npz and reused while it holds exactly these runs, columns and
    draws; anything else is computed again.
    """
    path = results_directory(root, "analysis") / "bootstrap_cache.npz"
    names = [f"{run_id}_seed{seed}" for run_id, seed in runs]
    # sha256 of every draw's positions end to end: a different test set, grouping, seed or number
    # of resamples gives a different digest, and the cache is not used.
    digest = hashlib.sha256(np.concatenate(draws).tobytes()).hexdigest()
    if path.exists():
        # np.load opens an .npz like a dict of arrays; the with block closes the file again.
        with np.load(path) as cached:
            if (cached["runs"].tolist() == names and cached["columns"].tolist() == list(CLASSIFICATION)
                    and str(cached["digest"]) == digest):
                values = cached["values"]
                return {key: values[position] for position, key in enumerate(runs)}
    values = []
    for position, key in enumerate(runs):
        # flush=True prints each line at once instead of when Python's output buffer fills.
        print(f"  bootstrap {key[0]} seed {key[1]} ({position + 1} of {len(runs)})", flush=True)
        values.append(bootstrap_rows(runs[key], draws))
    values = np.stack(values)
    # Written to a temporary name and renamed, so an interrupted run never leaves half a cache.
    temporary = path.with_name("bootstrap_cache.tmp.npz")
    np.savez_compressed(temporary, values=values, runs=np.array(names), columns=np.array(CLASSIFICATION),
                        digest=np.array(digest))
    temporary.replace(path)
    return {key: values[position] for position, key in enumerate(runs)}


def stability_arrays(run, positions):
    """Return one run's per-image stability values over the subset, as NumPy arrays.

    "in_family", "out_of_family" and "gap" are NaN outside the run's common set - images with no
    defined pass in a family - exactly as in the headline means. The "_zero" versions count every
    undefined pass (an empty or constant map) as 0 instead. "empty_in" / "empty_out" flag the
    images without a defined pass, and "correct" those the run classifies correctly at 0.5, from its
    test logits; positions gives each subset image's place in the test set.
    """
    records = run["records"]
    inside = np.array([family_score(record, "in") for record in records])
    outside = np.array([family_score(record, "out") for record in records])
    common = ~np.isnan(inside) & ~np.isnan(outside)
    # nan_to_num turns every NaN into 0.0, so an undefined pass counts as no agreement at all.
    zero = {family: np.array([np.mean(np.nan_to_num([item["pearson"] for item in record["passes"]
                                                      if item["family"] == family]))
                              for record in records])
            for family in ("in", "out")}
    predicted = classify(run["logits"])[positions]
    return {
        "in_family": np.where(common, inside, np.nan),
        "out_of_family": np.where(common, outside, np.nan),
        "gap": np.where(common, inside - outside, np.nan),
        "in_family_zero": zero["in"], "out_of_family_zero": zero["out"], "gap_zero": zero["in"] - zero["out"],
        "empty_in": np.isnan(inside), "empty_out": np.isnan(outside),
        "correct": predicted == run["labels"].numpy()[positions],
    }


def subset_weights(names, samples=BOOTSTRAP_SAMPLES, seed=SEED):
    """Return a (samples, images) array: how many times each subset image is in each resample.

    The stability subset is resampled by patient group, like everything else. A mean over one
    resample is then a mean weighted by these counts, so every stability difference on every
    resample is one matrix product - and every run is compared on the same resamples.
    """
    groups = patient_groups([Path(name) for name in names])
    rng = np.random.default_rng(seed)
    weights = np.zeros((samples, len(names)))
    for row in range(samples):
        drawn = rng.integers(0, len(groups), len(groups))
        # np.add.at adds 1 once per listed position, repeats included; weights[row][index] += 1
        # would count a position drawn twice only once.
        np.add.at(weights[row], np.concatenate([groups[position] for position in drawn]), 1)
    return weights


def interval(values):
    """Return the 95% percentile interval (low, high) of a statistic over the resamples."""
    tail = (100 - CONFIDENCE) / 2
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def paired_mean(difference, kept, weights):
    """Return (mean, per-resample means, images kept) of per-image differences over the kept images."""
    values = np.where(kept, difference, 0.0)
    # @ is matrix multiplication: each row of weights times values is one resample's weighted sum.
    return float(values.sum() / kept.sum()), (weights @ values) / (weights @ kept), int(kept.sum())


def adjusted(score, chance):
    """Return a Pearson score rescaled so the run's chance baseline becomes 0 and a perfect match 1."""
    return (score - chance) / (1 - chance)


def run_differences(first, second, runs, cache, arrays, weights, stability=True):
    """Return {(metric, variant): (difference, per-resample differences, images)}: run second minus first.

    Classification metrics come from the results files and the bootstrap cache. The validation
    metrics and the NORMAL images' margins (operating_point) are point estimates, with no interval:
    validation is the set the checkpoint and threshold were chosen on. Stability comes from the
    per-image arrays: the three measures in each variant, the share of empty maps per family, and
    the chance-adjusted measures - a point estimate only. stability=False leaves it out, for two
    runs of different architectures, which start from different gaps (NOTES.md, the reading rule).
    """
    results = {}
    for column, name in enumerate(CLASSIFICATION):
        results[name, ""] = (runs[second]["metrics"][name] - runs[first]["metrics"][name],
                             cache[second][:, column] - cache[first][:, column], len(runs[first]["images"]))
    for name in METRICS:
        results[f"validation_{name}", ""] = (runs[second]["validation"][name] - runs[first]["validation"][name],
                                             None, len(runs[first]["val_images"]))
    for name in ("normal_margin_median_val", "normal_margin_median_test"):
        results[name, ""] = (runs[second][name] - runs[first][name], None, None)
    if not stability:
        return results
    one, two = arrays[first], arrays[second]
    defined = ~np.isnan(one["gap"]) & ~np.isnan(two["gap"])
    everything = np.ones(len(defined), dtype=bool)
    kept = {"all": defined, "correct": defined & one["correct"] & two["correct"], "empty_zero": everything}
    for variant in VARIANTS:
        suffix = "_zero" if variant == "empty_zero" else ""
        for measure in STABILITY:
            results[f"stability_{measure}", variant] = paired_mean(
                two[measure + suffix] - one[measure + suffix], kept[variant], weights)
    for family in ("in", "out"):
        # Booleans become 0 and 1, so the mean difference is the change in the share of empty maps.
        change = two[f"empty_{family}"].astype(float) - one[f"empty_{family}"].astype(float)
        results[f"empty_maps_{family}", "all"] = paired_mean(change, everything, weights)
    for measure in STABILITY:
        rescaled = []
        for values, key in ((one, first), (two, second)):
            mean = values[measure][defined].mean()
            chance = runs[key]["metrics"]["stability"]["chance_baseline"]["pearson"]
            # A gap is a difference of two scores: the baseline cancels and only the scale remains.
            rescaled.append(mean / (1 - chance) if measure == "gap" else adjusted(mean, chance))
        results[f"stability_{measure}_adjusted", "all"] = (rescaled[1] - rescaled[0], None, int(defined.sum()))
    return results


def threshold_differences(key, runs, cache):
    """Return {(metric, ""): (difference, per-resample differences, images)}: tuned threshold minus 0.5."""
    metrics, rows = runs[key]["metrics"], cache[key]
    return {(name, ""): (metrics[f"{name}_tuned"] - metrics[name],
                         rows[:, CLASSIFICATION.index(f"{name}_tuned")] - rows[:, CLASSIFICATION.index(name)],
                         len(runs[key]["images"]))
            for name in METRICS[1:]}


def interaction(larger, smaller):
    """Return {(metric, variant): (difference, per-resample differences, images)}: one effect minus another.

    Both are run_differences results - the same contrast in two settings: within two architectures
    (say 100% minus 20% data, in the larger and the smaller model), or at two data sizes (say strong
    minus no augmentation, at the larger size and at 20%). Every run is scored on the same resamples,
    so the difference of the two effects is paired too: point minus point, and resample minus
    resample where both have resamples.
    """
    return {key: (larger[key][0] - smaller[key][0],
                  None if larger[key][1] is None else larger[key][1] - smaller[key][1], None)
            for key in larger}


def linear(key):
    """Return whether a (metric, variant) row adds up exactly across runs, so two routes to it must agree.

    A classification, validation or margin row is one number per run. A stability row with an empty map
    scored 0, and the share of empty maps, average over the same 468 images in every run. The other
    stability rows keep only the images defined in both runs of a pair, so their differences need not
    add up.
    """
    metric, variant = key
    return variant in ("", "empty_zero") or metric.startswith("empty_maps")


def two_ways(rows):
    """Return the largest disagreement between each regime-by-data-size row and the data-size rows it must equal.

    A regime-by-data-size row is the regime effect at the larger size minus the same effect at 20%. By
    arithmetic that is the augmented runs' change with more data minus the unaugmented runs' - two rows
    that pairs() made on its own, found here by their labels. A new row that paired the wrong runs would
    therefore disagree with them. Checked on the linear rows, in every view (NOTES.md #35); expected 0 up
    to float rounding. (Computing both sides from the same runs would not do: the identity holds for any
    four runs, so it could never catch a wrong pairing.)
    """
    changes = {(row["model"], row["contrast"], row["augmentation"], row["metric"], row["variant"], row["view"]):
               row["difference"] for row in rows if row["kind"] == "data size"}
    gaps = [0.0]
    for row in rows:
        if row["kind"] == "regime by data size" and linear((row["metric"], row["variant"])):
            # contrast is "<augmented regime>-none"; split("-") cuts it into the two regime names.
            augmented, unaugmented = row["contrast"].split("-")
            key = (row["model"], row["data_size"])
            rest = (row["metric"], row["variant"], row["view"])
            expected = changes[key + (LOG_LABELS[augmented],) + rest] - changes[key + (LOG_LABELS[unaugmented],) + rest]
            gaps.append(abs(row["difference"] - expected))
    return max(gaps)


def views(results):
    """Return the view rows of one comparison and metric, from {view: (difference, draws, images)}.

    One row per seed as given. With two seeds, a "mean" row follows: the average difference with its
    interval from the same resamples - test-sampling noise for two fixed trained models, not
    training noise - flagged "consistent" when both seeds agree in sign, and "strict" when both
    intervals also exclude 0.
    """
    rows = []
    for view, (difference, draws, images) in results.items():
        low, high = interval(draws) if draws is not None else (None, None)
        rows.append({"view": view, "difference": difference, "low": low, "high": high, "images": images,
                     "consistent": None, "strict": None})
    if len(rows) == 2:
        (_, (_, first_draws, _)), (_, (_, second_draws, _)) = results.items()
        one, two = rows
        mean_draws = (first_draws + second_draws) / 2 if first_draws is not None else None
        low, high = interval(mean_draws) if mean_draws is not None else (None, None)
        # np.sign is -1, 0 or +1; a difference of exactly 0 agrees with nothing.
        consistent = bool(np.sign(one["difference"]) == np.sign(two["difference"]) != 0)
        strict = None
        if one["low"] is not None:
            strict = consistent and ((one["low"] > 0 and two["low"] > 0) or (one["high"] < 0 and two["high"] < 0))
        rows.append({"view": "mean", "difference": (one["difference"] + two["difference"]) / 2, "low": low,
                     "high": high, "images": None, "consistent": consistent, "strict": strict})
    return rows


def experiment_ids():
    """Return {(model, split, augmentation): experiment ID}, to name a comparison by what differs."""
    return {(config.model, config.split, config.augmentation): run_id for run_id, config in EXPERIMENTS.items()}


def pairs():
    """Yield (kind, contrast, fixed columns, first ID, second ID) for every two-experiment comparison.

    Regime: each augmented regime against no augmentation, and strong against mild, within one
    architecture and data size. Data size: 50% and 100% against 20%, within one architecture and
    regime (NOTES.md, how the stability results are read).
    """
    ids = experiment_ids()
    for model in MODELS:
        for split in TRAIN_SPLITS:
            for second, first in REGIME_CONTRASTS:
                fixed = {"model": LOG_LABELS[model], "data_size": LOG_LABELS[split], "augmentation": ""}
                yield "regime", f"{second}-{first}", fixed, ids[model, split, first], ids[model, split, second]
        for augmentation in AUGMENTATIONS:
            for second, first in SIZE_CONTRASTS:
                fixed = {"model": LOG_LABELS[model], "data_size": "", "augmentation": LOG_LABELS[augmentation]}
                yield ("data size", f"{LOG_LABELS[second]}-{LOG_LABELS[first]}", fixed,
                       ids[model, first, augmentation], ids[model, second, augmentation])


def architecture_pairs():
    """Yield (contrast, fixed columns, first ID, second ID): a larger model against a smaller one in each cell.

    Levels only - which architecture scores higher at one data size and regime. Whether the
    architectures respond differently to data or augmentation is interaction_pairs.
    """
    ids = experiment_ids()
    for larger, smaller in ARCHITECTURE_CONTRASTS:
        for split in TRAIN_SPLITS:
            for augmentation in AUGMENTATIONS:
                fixed = {"model": "", "data_size": LOG_LABELS[split], "augmentation": LOG_LABELS[augmentation]}
                yield (f"{LOG_LABELS[larger]} minus {LOG_LABELS[smaller]}", fixed,
                       ids[smaller, split, augmentation], ids[larger, split, augmentation])


def interaction_pairs():
    """Yield (contrast, fixed columns, smaller model's pair, larger model's pair) for every interaction.

    The same within-architecture effect - mild or strong augmentation against none at one data size,
    or 50% or 100% of the data against 20% in one regime - in a larger and a smaller architecture. The
    interaction is the larger model's effect minus the smaller model's: positive means the larger
    model gains more (the proposal's third question).
    """
    effects = {}
    for kind, contrast, fixed, first, second in pairs():
        if contrast != "strong-mild":
            # What the effect holds fixed besides the model: the data size of a regime effect, the
            # regime of a data-size effect.
            effects[fixed["model"], contrast, fixed["data_size"] or fixed["augmentation"]] = (first, second)
    for larger, smaller in ARCHITECTURE_CONTRASTS:
        for (model, contrast, held), smaller_pair in effects.items():
            if model == LOG_LABELS[smaller]:
                regime_effect = held in (LOG_LABELS[split] for split in TRAIN_SPLITS)
                fixed = {"model": f"{LOG_LABELS[larger]} minus {LOG_LABELS[smaller]}",
                         "data_size": held if regime_effect else "", "augmentation": "" if regime_effect else held}
                yield contrast, fixed, smaller_pair, effects[LOG_LABELS[larger], contrast, held]


def regime_size_pairs():
    """Yield (contrast, fixed columns, effect at 20%, effect at the larger size) for the regime by data size.

    An effect is a pair of experiment IDs, (unaugmented, augmented), of one architecture at one data
    size. The comparison is the effect at 50% or 100% minus the same effect at 20% (NOTES.md #35), so
    the data_size column holds two sizes, e.g. "100%-20%".
    """
    ids = experiment_ids()
    for model in MODELS:
        for second, first in REGIME_CONTRASTS[:2]:
            for larger, smaller in SIZE_CONTRASTS:
                fixed = {"model": LOG_LABELS[model], "data_size": f"{LOG_LABELS[larger]}-{LOG_LABELS[smaller]}",
                         "augmentation": ""}
                yield (f"{second}-{first}", fixed, (ids[model, smaller, first], ids[model, smaller, second]),
                       (ids[model, larger, first], ids[model, larger, second]))


def comparisons(runs, cache, arrays, weights):
    """Return every comparison as rows for comparisons.csv, one per comparison, metric and view.

    Regime, data size, threshold, architecture levels, interactions and the regime by data size are
    compared within each seed and then across the two seeds; the seed comparison is each experiment's
    second seed minus its first. The kinds added later come last, so the rows of the others keep their
    order.
    """
    rows = []

    def add(kind, contrast, fixed, first, second, per_view):
        """Append the rows of one comparison: every metric it has, in every view."""
        # next(iter(...)) takes the first view's results, just to list the metrics they hold.
        for metric, variant in next(iter(per_view.values())):
            for view in views({name: results[metric, variant] for name, results in per_view.items()}):
                rows.append({"kind": kind, "contrast": contrast, **fixed, "first": first, "second": second,
                             "metric": metric, "variant": variant, **view})

    for kind, contrast, fixed, first, second in pairs():
        add(kind, contrast, fixed, first, second,
            {f"seed {seed}": run_differences((first, seed), (second, seed), runs, cache, arrays, weights)
             for seed in SEEDS})
    for run_id, config in EXPERIMENTS.items():
        fixed = {"model": LOG_LABELS[config.model], "data_size": LOG_LABELS[config.split],
                 "augmentation": LOG_LABELS[config.augmentation]}
        add("threshold", "tuned-0.5", fixed, run_id, run_id,
            {f"seed {seed}": threshold_differences((run_id, seed), runs, cache) for seed in SEEDS})
        add("seed", f"{SEEDS[1]}-{SEEDS[0]}", fixed, run_id, run_id,
            {f"{SEEDS[1]}-{SEEDS[0]}": run_differences((run_id, SEEDS[0]), (run_id, SEEDS[1]), runs, cache,
                                                       arrays, weights)})
    for contrast, fixed, first, second in architecture_pairs():
        add("architecture", contrast, fixed, first, second,
            {f"seed {seed}": run_differences((first, seed), (second, seed), runs, cache, arrays, weights,
                                             stability=False)
             for seed in SEEDS})
    for contrast, fixed, smaller, larger in interaction_pairs():
        add("interaction", contrast, fixed, "->".join(smaller), "->".join(larger),
            {f"seed {seed}": interaction(
                run_differences((larger[0], seed), (larger[1], seed), runs, cache, arrays, weights),
                run_differences((smaller[0], seed), (smaller[1], seed), runs, cache, arrays, weights))
             for seed in SEEDS})
    for contrast, fixed, smaller, larger in regime_size_pairs():
        add("regime by data size", contrast, fixed, "->".join(smaller), "->".join(larger),
            {f"seed {seed}": interaction(
                run_differences((larger[0], seed), (larger[1], seed), runs, cache, arrays, weights),
                run_differences((smaller[0], seed), (smaller[1], seed), runs, cache, arrays, weights))
             for seed in SEEDS})
    return rows


def run_rows(runs):
    """Return one descriptive row per run for runs.csv: errors, stability, empty maps, flips, operating point.

    A flip is a perturbed pass whose prediction differs from the clean image's, at the run's tuned
    threshold (as recorded by src/stability.py), counted per pass and split by direction. The last
    columns serve NOTES.md #33: validation metrics, the NORMAL images' margins, how long the run
    trained, and - a diagnostic only, never a result - balanced accuracy at a threshold chosen on the
    test set itself, which shows whether a good cut-off still exists on test.
    """
    rows = []
    for (run_id, seed), run in runs.items():
        metrics, stability = run["metrics"], run["metrics"]["stability"]
        chance = stability["chance_baseline"]["pearson"]
        row = {"experiment": run_id, "seed": seed, "model": metrics["model"], "data_size": metrics["data_size"],
               "augmentation": metrics["augmentation"], "threshold": metrics["threshold"]}
        row.update({name: metrics[name] for name in ("roc_auc", "balanced_accuracy", "balanced_accuracy_tuned",
                                                      "recall", "specificity", "recall_tuned", "specificity_tuned")})
        row.update({"false_alarms": metrics["confusion_matrix"]["fp"], "misses": metrics["confusion_matrix"]["fn"],
                    "false_alarms_tuned": metrics["confusion_matrix_tuned"]["fp"],
                    "misses_tuned": metrics["confusion_matrix_tuned"]["fn"]})
        inside, outside = metrics["gradcam_stability_in_family"], metrics["gradcam_stability_out_of_family"]
        row.update({"stability_in_family": inside, "stability_out_of_family": outside,
                    "stability_gap": metrics["gradcam_stability_gap"], "chance_baseline": chance,
                    "stability_in_family_adjusted": adjusted(inside, chance),
                    "stability_out_of_family_adjusted": adjusted(outside, chance),
                    "stability_gap_adjusted": metrics["gradcam_stability_gap"] / (1 - chance),
                    "empty_maps_in": stability["excluded_in_family"], "empty_maps_out": stability["excluded_out_of_family"]})
        for family, name in (("in", "rotation"), ("out", "noise")):
            passes = [(record["predicted"], item) for record in run["records"]
                      for item in record["passes"] if item["family"] == family]
            row[f"{name}_flips_to_pneumonia"] = sum(item["flipped"] and clean == "NORMAL" for clean, item in passes)
            row[f"{name}_flips_to_normal"] = sum(item["flipped"] and clean == "PNEUMONIA" for clean, item in passes)
        row.update({f"validation_{name}": run["validation"][name]
                    for name in ("roc_auc", "balanced_accuracy", "recall", "specificity")})
        row.update({name: run[name] for name in ("normal_margin_median_val", "normal_margin_p90_val",
                                                 "normal_margin_median_test", "normal_margin_p90_test")})
        # The epoch the checkpoint holds: the best on the selection metric, as src/train.py kept it.
        best = max(run["history"], key=lambda epoch: epoch[RECIPE.selection_metric])
        # A diagnostic only: this threshold is chosen on the test set, so no result may use it.
        test_threshold = choose_threshold(run["logits"], run["labels"])
        row.update({"epochs_trained": len(run["history"]), "best_epoch": best["epoch"],
                    "diagnostic_balanced_accuracy_test_threshold":
                        scores(run["logits"], run["labels"], test_threshold, with_auc=False)["balanced_accuracy"]})
        rows.append(row)
    return rows


def checks(runs, cache, arrays):
    """Return the largest disagreement with the results files, per kind of number the analysis uses.

    Every difference is built from these pieces, so each must reproduce what the results files say:
    the metrics from the re-scored logits, the stability means from the per-image records, the
    seed-42 intervals - because the draws are the ones evaluate.py made with seed 42 - and the
    validation metrics from the re-scored validation logits, against validation_metrics.json.
    """
    worst = {"metrics": 0.0, "stability": 0.0, "seed-42 intervals": 0.0, "validation metrics": 0.0}
    for key, run in runs.items():
        worst["validation metrics"] = max([worst["validation metrics"]] + [
            abs(run["validation"][name] - run["stored_validation"][name]) for name in METRICS])
        metrics = run["metrics"]
        fresh = scores(run["logits"], run["labels"])
        fresh.update({f"{name}_tuned": value for name, value in
                      scores(run["logits"], run["labels"], metrics["threshold_logit_margin"], with_auc=False).items()})
        worst["metrics"] = max([worst["metrics"]] + [abs(value - metrics[name]) for name, value in fresh.items()])
        worst["stability"] = max([worst["stability"]] + [
            abs(np.nanmean(arrays[key][measure]) - metrics[f"gradcam_stability_{measure}"]) for measure in STABILITY])
        if key[1] == SEED:
            for column, name in enumerate(CLASSIFICATION):
                low, high = interval(cache[key][:, column])
                worst["seed-42 intervals"] = max(worst["seed-42 intervals"], abs(low - metrics[f"{name}_ci_low"]),
                                                 abs(high - metrics[f"{name}_ci_high"]))
    return worst


def select(rows, **fields):
    """Return the rows whose columns equal every value given, e.g. select(rows, kind="seed")."""
    return [row for row in rows if all(row[name] == value for name, value in fields.items())]


def pattern_lines(rows, groups):
    """Return markdown table lines: per group and measure, how often each seed's difference is positive.

    groups is [(label, filter)]; each filter selects a set of cells (e.g. mild-none in all nine
    architecture x size cells), and the columns count them in each seed view.
    """
    lines = ["| Comparison | Measure | Seed 42 > 0 | Seed 43 > 0 | Consistent + / - | Strict + / - | Median mean |",
             "|---|---|---|---|---|---|---|"]
    for label, filters in groups:
        for metric, variant, name in MEASURES:
            cells = {view: select(rows, **filters, metric=metric, variant=variant, view=view)
                     for view in (f"seed {SEEDS[0]}", f"seed {SEEDS[1]}", "mean")}
            total = len(cells["mean"])
            positive = [sum(row["difference"] > 0 for row in cells[f"seed {seed}"]) for seed in SEEDS]
            counts = {}
            for flag in ("consistent", "strict"):
                # A flag holds in one direction: count the cells above 0 and below 0 apart.
                counts[flag] = [sum(bool(row[flag]) and np.sign(row["difference"]) == sign for row in cells["mean"])
                                for sign in (1, -1)]
            median = np.median([row["difference"] for row in cells["mean"]])
            lines.append(f"| {label} | {name} | {positive[0]}/{total} | {positive[1]}/{total} | "
                         f"{counts['consistent'][0]} / {counts['consistent'][1]} | "
                         f"{counts['strict'][0]} / {counts['strict'][1]} | {median:+.4f} |")
    return lines


def direction_lines(rows, groups, measures):
    """Return markdown table lines like pattern_lines, with ties counted: up / equal / down per seed.

    Strict needs intervals, so it shows "-" for measures that have none (validation, margins). The
    tables written before 2026-09-28 keep pattern_lines, so they stay exactly as they were.
    """
    lines = ["| Comparison | Measure | Seed 42 up / equal / down | Seed 43 up / equal / down | Consistent + / - | "
             "Strict + / - | Median mean |", "|---|---|---|---|---|---|---|"]
    for label, filters in groups:
        for metric, variant, name in measures:
            cells = {view: select(rows, **filters, metric=metric, variant=variant, view=view)
                     for view in (f"seed {SEEDS[0]}", f"seed {SEEDS[1]}", "mean")}
            per_seed = []
            for seed in SEEDS:
                # np.sign gives 1, 0 or -1; list.count then counts each direction.
                signs = [np.sign(row["difference"]) for row in cells[f"seed {seed}"]]
                per_seed.append(f"{signs.count(1)} / {signs.count(0)} / {signs.count(-1)}")
            counts = {flag: " / ".join(str(sum(bool(row[flag]) and np.sign(row["difference"]) == sign
                                               for row in cells["mean"])) for sign in (1, -1))
                      for flag in ("consistent", "strict")}
            if all(row["strict"] is None for row in cells["mean"]):
                counts["strict"] = "-"
            median = np.median([row["difference"] for row in cells["mean"]])
            lines.append(f"| {label} | {name} | {per_seed[0]} | {per_seed[1]} | {counts['consistent']} | "
                         f"{counts['strict']} | {median:+.4f} |")
    return lines


def summary(rows, per_run, worst, samples):
    """Return summary.md: the checks, then the patterns the report can draw on, as markdown tables."""
    lines = [
        "# Results analysis - both seeds",
        "",
        "Written by `python -m src.analysis` from `results/`; change the code, not this file. Every difference",
        "is second minus first, paired on the same test images, with a 95% interval from "
        f"{samples} patient-group resamples",
        "shared by every run. The intervals cover test-sampling noise for fixed trained models, not training",
        "noise - the seed rows show that. No comparison is primary: the tables count how often a pattern holds",
        "(NOTES.md, how the stability results are read); the report gives the two seeds' mean and calls a finding",
        "strong only when both seeds agree (NOTES.md #32). All the numbers: `comparisons.csv` and `runs.csv`.",
        "",
        "**Views.** Seed 42 and seed 43 are each seed's own paired difference. *Consistent*: both seeds agree in",
        "sign (+ both above 0, - both below). *Strict*: consistent, and both intervals exclude 0. *Median mean*:",
        "the median over the cells of the two seeds' mean difference.",
        "",
        "## Checks against the results files",
        "",
        "Largest absolute difference between what the analysis recomputes and what the results files hold:",
        "",
    ]
    lines += [f"- {name}: {value:.1e}" + ("" if value <= EXACT else " - **does not match**")
              for name, value in worst.items()]

    lines += ["", "## Augmentation regime, within architecture and data size (9 cells each)", ""]
    lines += pattern_lines(rows, [(contrast, {"kind": "regime", "contrast": contrast})
                                  for contrast in (f"{second}-{first}" for second, first in REGIME_CONTRASTS)])

    lines += ["", "## Data size, within architecture and regime (3 cells each)", ""]
    lines += pattern_lines(rows, [(f"{contrast}, {LOG_LABELS[regime]}",
                                   {"kind": "data size", "contrast": contrast, "augmentation": LOG_LABELS[regime]})
                                  for contrast in (f"{LOG_LABELS[second]}-{LOG_LABELS[first]}"
                                                   for second, first in SIZE_CONTRASTS)
                                  for regime in AUGMENTATIONS])

    lines += ["", "## Chance-adjusted stability: does rescaling change the sign? (regime, 9 cells each)", "",
              "| Comparison | Seed | In-family agrees | Out-of-family agrees | Gap agrees |", "|---|---|---|---|---|"]
    for second, first in REGIME_CONTRASTS:
        for seed in SEEDS:
            agree = []
            for measure in STABILITY:
                raw = select(rows, kind="regime", contrast=f"{second}-{first}", metric=f"stability_{measure}",
                             variant="all", view=f"seed {seed}")
                scaled = select(rows, kind="regime", contrast=f"{second}-{first}",
                                metric=f"stability_{measure}_adjusted", variant="all", view=f"seed {seed}")
                agree.append(sum(np.sign(a["difference"]) == np.sign(b["difference"]) for a, b in zip(raw, scaled)))
            lines.append(f"| {second}-{first} | {seed} | {agree[0]}/9 | {agree[1]}/9 | {agree[2]}/9 |")

    lines += ["", f"## Seed yardstick: seed {SEEDS[1]} minus seed {SEEDS[0]}, per experiment (27)", "",
              "| Measure | Median abs | 90th percentile abs | Largest abs | Intervals excluding 0 |", "|---|---|---|---|---|"]
    for metric, variant, name in MEASURES:
        cells = select(rows, kind="seed", metric=metric, variant=variant)
        sizes = np.abs([row["difference"] for row in cells])
        outside = sum(row["low"] > 0 or row["high"] < 0 for row in cells)
        lines.append(f"| {name} | {np.median(sizes):.4f} | {np.percentile(sizes, 90):.4f} | {sizes.max():.4f} | "
                     f"{outside}/{len(cells)} |")

    lines += ["", "## Tuned threshold minus 0.5, per experiment (27)", "",
              "| Metric | Seed | Better | Worse | Same | Mean difference |", "|---|---|---|---|---|---|"]
    for metric in ("balanced_accuracy", "specificity", "recall", "f1"):
        for seed in SEEDS:
            values = [row["difference"] for row in select(rows, kind="threshold", metric=metric, view=f"seed {seed}")]
            lines.append(f"| {metric.replace('_', ' ')} | {seed} | {sum(v > 0 for v in values)} | "
                         f"{sum(v < 0 for v in values)} | {sum(v == 0 for v in values)} | {np.mean(values):+.4f} |")

    lines += ["", "## Errors and prediction flips, summed over the 27 experiments", "",
              "| Seed | False alarms @0.5 | Misses @0.5 | False alarms, tuned | Misses, tuned | "
              "Rotation flips to PNEUMONIA / to NORMAL | Noise flips to PNEUMONIA / to NORMAL |",
              "|---|---|---|---|---|---|---|"]
    for seed in SEEDS:
        own = [row for row in per_run if row["seed"] == seed]
        total = {name: sum(row[name] for row in own) for name in own[0] if isinstance(own[0][name], int)
                 and name != "seed"}
        lines.append(f"| {seed} | {total['false_alarms']} | {total['misses']} | {total['false_alarms_tuned']} | "
                     f"{total['misses_tuned']} | {total['rotation_flips_to_pneumonia']} / "
                     f"{total['rotation_flips_to_normal']} | {total['noise_flips_to_pneumonia']} / "
                     f"{total['noise_flips_to_normal']} |")

    lines += ["", "## Empty maps under noise, per architecture (of 468 subset images)", "",
              "| Architecture | Seed | Fewest | Most | Runs above 10% |", "|---|---|---|---|---|"]
    for model in MODELS:
        for seed in SEEDS:
            counts = [row["empty_maps_out"] for row in per_run if row["seed"] == seed and row["model"] == LOG_LABELS[model]]
            lines.append(f"| {LOG_LABELS[model]} | {seed} | {min(counts)} | {max(counts)} | "
                         f"{sum(count > 0.1 * 468 for count in counts)}/{len(counts)} |")
    return "\n".join(lines + follow_up_sections(rows, per_run) + size_section(rows)) + "\n"


def follow_up_sections(rows, per_run):
    """Return the summary.md sections added on 2026-09-28: NOTES.md #33, and lightweight against larger models."""
    size_contrasts = [f"{LOG_LABELS[second]}-{LOG_LABELS[first]}" for second, first in SIZE_CONTRASTS]
    regime_effects = [f"{second}-{first}" for second, first in REGIME_CONTRASTS[:2]]
    architectures = [f"{LOG_LABELS[larger]} minus {LOG_LABELS[smaller]}" for larger, smaller in ARCHITECTURE_CONTRASTS]

    lines = ["", "## Data size and the operating point (NOTES.md #33; 9 cells each)", "",
             "The validation rows and the NORMAL images' margins are point estimates, without intervals: validation is",
             "the set the checkpoint and the threshold were chosen on, so its values are optimistic, and paired",
             "validation intervals would need a second cache. The margin is the PNEUMONIA logit minus the NORMAL logit;",
             "a model says PNEUMONIA above 0 (the 0.5 threshold), so NORMAL margins rising towards 0 mean more false",
             "alarms at any fixed threshold. Specificity counts NORMAL images only, so the class balance of the two",
             "sets (validation 202 / 583, test 234 / 390) cannot explain a change in it.", ""]
    lines += direction_lines(rows, [(contrast, {"kind": "data size", "contrast": contrast})
                                    for contrast in size_contrasts], SHIFT_MEASURES)
    epochs = {(row["experiment"], row["seed"]): row["epochs_trained"] for row in per_run}
    size_pairs = [(first, second) for kind, _, _, first, second in pairs() if kind == "data size"]
    counts = ["seed {} {}".format(seed, " / ".join(
        str(sum(np.sign(epochs[second, seed] - epochs[first, seed]) == sign for first, second in size_pairs))
        for sign in (1, 0, -1))) for seed in SEEDS]
    diagnostic = [row["diagnostic_balanced_accuracy_test_threshold"] for row in per_run]
    at_half = [row["balanced_accuracy"] for row in per_run]
    lines += ["", f"Epochs trained, more data against less (up / equal / down of 18): {'; '.join(counts)}.", "",
              "**Diagnostic only, never a result** - the threshold is chosen on the test set itself: balanced accuracy "
              f"there is {min(diagnostic):.3f}-{max(diagnostic):.3f} over the 54 runs, against "
              f"{min(at_half):.3f}-{max(at_half):.3f} at 0.5."]

    lines += ["", "## Architecture levels: larger minus smaller, within data size and regime (3 cells each)", "",
              "Classification only: two architectures start from different stability gaps (NOTES.md, the reading rule).",
              ""]
    lines += direction_lines(rows, [(f"{contrast}, {LOG_LABELS[split]}",
                                     {"kind": "architecture", "contrast": contrast, "data_size": LOG_LABELS[split]})
                                    for contrast in architectures for split in TRAIN_SPLITS], LEVEL_MEASURES)

    lines += ["", "## Interactions: the larger model's effect minus the smaller model's (3 cells each)", "",
              "Positive: the larger model gains more from augmentation, or from more data, than the smaller one - the",
              "proposal's question whether lightweight and larger models behave differently under limited data.", ""]
    lines += direction_lines(rows, [(f"{contrast}, {pair}", {"kind": "interaction", "model": pair, "contrast": contrast})
                                    for pair in architectures for contrast in regime_effects + size_contrasts],
                             KEY_MEASURES)

    lines += ["", "## Per architecture: what augmentation and more data do (3 cells each)", ""]
    lines += direction_lines(rows, [(f"{LOG_LABELS[model]}, {contrast}",
                                     {"kind": kind, "model": LOG_LABELS[model], "contrast": contrast})
                                    for model in MODELS
                                    for kind, contrast in [("regime", effect) for effect in regime_effects]
                                    + [("data size", size) for size in size_contrasts]], KEY_MEASURES)
    return lines


def listing_lines(rows):
    """Return markdown table lines listing every regime-by-data-size comparison, one line per measure.

    Each line gives both seeds with their intervals, the mean and its two flags, and beside them the
    augmented and the unaugmented runs' own change with more data (the mean of the seeds, from the
    data-size rows). On the linear rows the comparison is exactly the first change minus the second,
    so the two show which side moved.
    """
    def shown(row):
        """Return a difference with its interval, or alone where it has none."""
        if row["low"] is None:
            return f"{row['difference']:+.4f}"
        return f"{row['difference']:+.4f} [{row['low']:+.4f}, {row['high']:+.4f}]"

    def flag(value):
        """Return a consistent or strict flag as yes / no, or "-" where it does not apply."""
        return "-" if value is None else ("yes" if value else "no")

    lines = ["| Architecture | Effect | Sizes | Measure | Seed 42 | Seed 43 | Mean | Consistent | Strict | "
             "Augmented runs' change | Unaugmented runs' change |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for contrast, fixed, _, _ in regime_size_pairs():
        # contrast is "<augmented regime>-none"; split("-") cuts it into the two regime names.
        augmented, unaugmented = contrast.split("-")
        for metric, variant, name in SIZE_MEASURES:
            cells = {row["view"]: row for row in select(rows, kind="regime by data size", **fixed, contrast=contrast,
                                                        metric=metric, variant=variant)}
            changes = [select(rows, kind="data size", model=fixed["model"], contrast=fixed["data_size"],
                              augmentation=LOG_LABELS[regime], metric=metric, variant=variant,
                              view="mean")[0]["difference"] for regime in (augmented, unaugmented)]
            lines.append(f"| {fixed['model']} | {contrast} | {fixed['data_size']} | {name} | "
                         f"{shown(cells[f'seed {SEEDS[0]}'])} | {shown(cells[f'seed {SEEDS[1]}'])} | "
                         f"{shown(cells['mean'])} | {flag(cells['mean']['consistent'])} | "
                         f"{flag(cells['mean']['strict'])} | {changes[0]:+.4f} | {changes[1]:+.4f} |")
    return lines


def size_section(rows):
    """Return the summary.md section added for NOTES.md #35: when augmentation helps, by data size."""
    lines = ["", "## Augmentation by data size (NOTES.md #35)", "",
             "Each new comparison is the regime effect at the larger data size minus the same effect at 20%, within one",
             "architecture and paired on the same resamples. By arithmetic it is also the augmented runs' change with more",
             "data minus the unaugmented runs' change, so a positive value says only that the effect is larger at the",
             "larger size - it can come from the unaugmented runs losing more, and the last two columns of the second table",
             "show which side moved. ROC-AUC shows ranking; specificity @0.5 and the NORMAL images' median test margin show",
             "the operating point, where the shift of NOTES.md #33 enters; the stability gap is given on the images defined",
             "in both runs and with an empty map scored 0 - the two can differ, because the images kept change with data",
             "size, most for ResNet18. Tuned balanced accuracy is left out: its threshold does not carry over to test (#31,",
             "#33). With random signs, a 3-cell row comes out 3 / 0 or 0 / 3 about 1 time in 32, so about one such row in",
             "the first table is expected by chance, and counts compared across data sizes do not show when augmentation",
             "helps - only the paired rows of the second table do.", "",
             "**The regime effect at each data size** (3 cells each: the three architectures):", ""]
    lines += direction_lines(rows, [(f"{contrast}, {LOG_LABELS[split]}",
                                     {"kind": "regime", "contrast": contrast, "data_size": LOG_LABELS[split]})
                                    for contrast in (f"{second}-{first}" for second, first in REGIME_CONTRASTS[:2])
                                    for split in TRAIN_SPLITS], SIZE_MEASURES)
    lines += ["", "**The change from 20%, per architecture** - the effect at the larger size minus at 20%; the last two",
              "columns are the augmented and the unaugmented runs' own change with more data, the mean of the seeds:", ""]
    return lines + listing_lines(rows)


def difference_panels(panels, rows, metric, variant):
    """Draw mild-none and strong-none per data size, both seeds, on one panel per architecture."""
    sizes = [LOG_LABELS[split] for split in TRAIN_SPLITS]
    for panel, model in zip(panels, MODELS):
        for number, (second, first) in enumerate(REGIME_CONTRASTS[:2]):
            for seed_number, seed in enumerate(SEEDS):
                cells = {row["data_size"]: row for row in select(
                    rows, kind="regime", contrast=f"{second}-{first}", model=LOG_LABELS[model], metric=metric,
                    variant=variant, view=f"seed {seed}")}
                values = np.array([cells[size]["difference"] for size in sizes])
                errors = [values - [cells[size]["low"] for size in sizes],
                          np.array([cells[size]["high"] for size in sizes]) - values]
                # Four markers side by side at each data size, so no interval hides another.
                x = np.arange(len(sizes)) + (2 * number + seed_number - 1.5) * 0.1
                colour = REGIME_COLOURS[second]
                # The seed is the marker's fill: solid for the first seed, hollow for the second.
                panel.errorbar(x, values, yerr=errors, color=colour, marker=REGIME_MARKERS[second], markersize=5,
                               linestyle="none", capsize=2, markerfacecolor=colour if seed_number == 0 else "white",
                               label=f"{second} - {first}, seed {seed}")
        panel.axhline(0, color="#6b6b68", linewidth=1)
        panel.set_xticks(range(len(sizes)), sizes)
        panel.set_title(LOG_LABELS[model], fontsize=9)
        panel.grid(alpha=0.25, linewidth=0.6)
        panel.set_axisbelow(True)


def difference_figure(rows, metric, variants, ylabel, title, path):
    """Draw regime differences with intervals: one row of architecture panels per variant."""
    figure = Figure(figsize=(9.6, 1.2 + 2.7 * len(variants)), layout="constrained")
    # squeeze=False keeps a 2D grid of axes even for a single row.
    grid = figure.subplots(len(variants), len(MODELS), sharey="row", squeeze=False)
    for panels, (variant, label) in zip(grid, variants):
        difference_panels(panels, rows, metric, variant)
        panels[0].set_ylabel(f"{ylabel}\n{label}" if label else ylabel, fontsize=8)
    for panel in grid[-1]:
        panel.set_xlabel("training data")
    figure.legend(*grid[0][0].get_legend_handles_labels(), ncols=4, fontsize=7, frameon=False,
                  loc="outside lower center")
    titled(figure, title)
    return save_figure(figure, path, title)


def flips_figure(per_run, path):
    """Draw per experiment how many perturbed passes flip the prediction, and in which direction."""
    ids = list(EXPERIMENTS)
    figure = Figure(figsize=(11.0, 6.4), layout="constrained")
    panels = figure.subplots(2, 1, sharex=True)
    for panel, name in zip(panels, ("rotation", "noise")):
        for seed_number, seed in enumerate(SEEDS):
            own = {row["experiment"]: row for row in per_run if row["seed"] == seed}
            x = np.arange(len(ids)) + (seed_number - 0.5) * 0.4
            colours = [REGIME_COLOURS[EXPERIMENTS[run_id].augmentation] for run_id in ids]
            # The second seed is hatched, so the two seeds stay apart without a second colour code.
            hatch = None if seed_number == 0 else "////"
            panel.bar(x, [own[run_id][f"{name}_flips_to_pneumonia"] for run_id in ids], width=0.4,
                      color=colours, hatch=hatch, edgecolor="white", linewidth=0)
            # Flips to NORMAL go below the axis, so the two directions read at a glance.
            panel.bar(x, [-own[run_id][f"{name}_flips_to_normal"] for run_id in ids], width=0.4,
                      color=colours, hatch=hatch, edgecolor="white", linewidth=0)
        panel.axhline(0, color="#1a1a19", linewidth=0.8)
        for boundary in (8.5, 17.5):
            panel.axvline(boundary, color="#6b6b68", linewidth=0.8, linestyle="--")
        panel.set_ylabel(f"{name}: passes flipped\nup: to PNEUMONIA, down: to NORMAL", fontsize=8)
        panel.grid(axis="y", alpha=0.25, linewidth=0.6)
        panel.set_axisbelow(True)
    panels[-1].set_xticks(range(len(ids)), ids, rotation=90, fontsize=7)
    handles = [Patch(color=REGIME_COLOURS[regime], label=f"augmentation {regime}") for regime in AUGMENTATIONS]
    handles += [Patch(facecolor="#6b6b68", label=f"seed {SEEDS[0]}"),
                Patch(facecolor="#6b6b68", hatch="////", edgecolor="white", label=f"seed {SEEDS[1]}")]
    figure.legend(handles=handles, ncols=5, fontsize=8, frameon=False, loc="outside lower center")
    title = ("Prediction flips on the stability subset (2 passes per image and family), by direction - "
             "MobileNetV3-Small | ResNet18 | EfficientNet-B0")
    titled(figure, title)
    return save_figure(figure, path, title)


def operating_point_figure(per_run, splits, title, path):
    """Draw the NORMAL images' median margin against data size: one row of architecture panels per split.

    splits is [(split, name)], e.g. [("val", "validation"), ("test", "test")]; each panel has a line
    per regime and seed. The margin needs no threshold: above 0 an image is called PNEUMONIA at 0.5,
    and every row draws that line - the test row shows how close the NORMAL images come to false
    alarms, the validation row how far below it they sit where the checkpoint was chosen (NOTES.md #33).
    """
    sizes = [LOG_LABELS[split] for split in TRAIN_SPLITS]
    figure = Figure(figsize=(9.6, 0.6 + 3.0 * len(splits)), layout="constrained")
    # squeeze=False keeps the grid two-dimensional even with one row, so grid[row][column] always works.
    grid = figure.subplots(len(splits), len(MODELS), sharey="row", squeeze=False)
    for panels, (split, name) in zip(grid, splits):
        for panel, model in zip(panels, MODELS):
            for regime in AUGMENTATIONS:
                for seed_number, seed in enumerate(SEEDS):
                    own = {row["data_size"]: row for row in per_run if row["seed"] == seed
                           and row["model"] == LOG_LABELS[model] and row["augmentation"] == LOG_LABELS[regime]}
                    colour = REGIME_COLOURS[regime]
                    # The seed is the marker's fill and the line style: solid for the first, hollow and dashed for the second.
                    panel.plot(range(len(sizes)), [own[size][f"normal_margin_median_{split}"] for size in sizes],
                               color=colour, marker=REGIME_MARKERS[regime], markersize=5,
                               linestyle="-" if seed_number == 0 else "--",
                               markerfacecolor=colour if seed_number == 0 else "white",
                               label=f"{regime}, seed {seed}")
            # The default cut-off: a NORMAL image above this line is a false alarm at 0.5. axhline also
            # stretches the y-axis to take in 0, so a row far below the line shows how far.
            panel.axhline(0, color="#6b6b68", linewidth=1)
            panel.set_xticks(range(len(sizes)), sizes)
            panel.set_title(f"{LOG_LABELS[model]} - {name}", fontsize=9)
            panel.grid(alpha=0.25, linewidth=0.6)
            panel.set_axisbelow(True)
        # One label for every row - the panel titles name the split - at a fixed distance from the axis.
        # Left to itself, matplotlib sets it beside the tick labels, and "-18" is wider than "-4": the
        # panels of the two single-row files would sit 13 pixels apart when stacked in the report.
        panels[0].set_ylabel("NORMAL images' median margin\n(above 0: pneumonia at 0.5)", fontsize=8)
        # set_label_coords places the label in fractions of the panel: x = -0.16 is 16% of its width to the left.
        panels[0].yaxis.set_label_coords(-0.16, 0.5)
    for panel in grid[-1]:
        panel.set_xlabel("training data")
    figure.legend(*grid[0][0].get_legend_handles_labels(), ncols=6, fontsize=7, frameon=False,
                  loc="outside lower center")
    titled(figure, title)
    return save_figure(figure, path, title)


def write_csv(rows, path):
    """Write dict rows to a CSV file, with the first row's keys as the columns."""
    # newline="" is what the csv module expects: it writes its own line endings.
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse(root=None, samples=BOOTSTRAP_SAMPLES):
    """Compare every run from the archive and write results/analysis/; return the comparison rows."""
    runs = load_runs(root)
    first = next(iter(runs.values()))
    print(f"{len(runs)} runs; {len(first['images'])} test images, {len(first['records'])} in the stability subset")
    draws = test_draws(first["images"], first["labels"], samples)
    cache = bootstrap_cache(runs, draws, root)
    # Where each subset image sits in the test set, so stability can use the test predictions.
    position = {name: index for index, name in enumerate(first["images"])}
    names = [subset_name(record) for record in first["records"]]
    positions = np.array([position[name] for name in names])
    arrays = {key: stability_arrays(run, positions) for key, run in runs.items()}
    weights = subset_weights(names, samples)
    rows = comparisons(runs, cache, arrays, weights)
    per_run = run_rows(runs)
    worst = checks(runs, cache, arrays)
    # Not a check against the results files, but printed with them: the new rows against the data-size rows.
    worst["regime by data size, two ways (internal)"] = two_ways(rows)

    directory = results_directory(root, "analysis")
    write_csv(rows, directory / "comparisons.csv")
    write_csv(per_run, directory / "runs.csv")
    (directory / "summary.md").write_text(summary(rows, per_run, worst, len(draws)), encoding="utf-8")
    difference_figure(rows, "roc_auc", [("", "")], "test ROC-AUC difference (95% CI)",
                      "Augmentation against none: test ROC-AUC difference per architecture and data size, both seeds",
                      directory / "analysis_regime_roc_auc.png")
    difference_figure(rows, "stability_gap", [("all", "images defined in both runs"),
                                              ("correct", "images correct in both runs"),
                                              ("empty_zero", "all images, empty map = 0")],
                      "gap difference (95% CI)",
                      "Augmentation against none: stability gap difference (in-family minus out-of-family), "
                      "three ways of treating the images, both seeds",
                      directory / "analysis_regime_gap.png")
    flips_figure(per_run, directory / "analysis_flips.png")
    splits = [("val", "validation"), ("test", "test")]
    operating_point_figure(per_run, splits, "NORMAL images' median logit margin by data size - validation (top) "
                           "and test (bottom), both seeds", directory / "analysis_operating_point_shift.png")
    # Each row in a file of its own as well, so the report can stack them or use one alone.
    for split, name in splits:
        operating_point_figure(per_run, [(split, name)], f"NORMAL images' median logit margin by data size - "
                               f"{name}, both seeds", directory / f"analysis_operating_point_{name}.png")
    for name, value in worst.items():
        print(f"  check against the results files, {name}: largest difference {value:.1e}"
              + ("" if value <= EXACT else " - DOES NOT MATCH"))
    print(f"{len(rows)} comparison rows and {len(per_run)} run rows written to {directory}")
    return rows


def self_check():
    """Check the re-scoring gate and the comparisons on synthetic data."""
    generator = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 2, (200,), generator=generator)
    # float64 logits that rank the classes well but not perfectly, like a trained model's; float64
    # so that a drift of 1e-9 is representable at all.
    logits = torch.stack([torch.zeros(200, dtype=torch.float64),
                          labels * 2.0 - 1 + torch.randn(200, generator=generator, dtype=torch.float64)], dim=1)
    threshold = choose_threshold(logits, labels)
    stored = {**scores(logits, labels), "threshold_logit_margin": threshold,
              "confusion_matrix": confusion_counts(logits, labels),
              "confusion_matrix_tuned": confusion_counts(logits, labels, threshold)}
    stored.update({f"{name}_tuned": value
                   for name, value in scores(logits, labels, threshold, with_auc=False).items()})
    # Moving every PNEUMONIA logit by 1e-9 shifts all margins and the threshold together: no prediction changes.
    drifted = logits + torch.tensor([0.0, 1e-9], dtype=torch.float64)
    changed = logits.clone()
    # One NORMAL image pushed far into PNEUMONIA: its prediction flips at any threshold.
    changed[(labels == 0).nonzero()[0, 0], 1] = 50.0
    print(f"  identical logits give PASS-exact: {gate(stored, logits, labels, logits, labels)[0] == 'PASS-exact'}")
    print(f"  a drift of 1e-9 gives PASS-approx: {gate(stored, drifted, labels, drifted, labels)[0] == 'PASS-approx'}")
    print(f"  one changed prediction gives FAIL: {gate(stored, logits, labels, changed, labels)[0] == 'FAIL'}")
    print(f"  image names keep class and file: "
          f"{image_names([Path('test/NORMAL/IM-0001-0001.jpeg')]) == ['NORMAL/IM-0001-0001.jpeg']}")

    # The comparisons: 30 NORMAL patients with one image each, 15 PNEUMONIA patients with two.
    names = ([f"NORMAL/IM-{number:04d}-0001.jpeg" for number in range(30)]
             + [f"PNEUMONIA/person{number}_bacteria_{image}.jpeg" for number in range(15) for image in (1, 2)])
    truth = torch.tensor([0] * 30 + [1] * 30)
    rng = np.random.default_rng(0)

    def synthetic_run(separation, empty=()):
        """Return a run dict whose margins separate the classes by `separation`; `empty` images get empty noise maps."""
        margin = torch.tensor(rng.normal(truth.numpy() * separation - separation / 2, 1.0), dtype=torch.float32)
        run_logits = torch.stack([torch.zeros(60), margin], dim=1)
        metrics = {**scores(run_logits, truth), "threshold_logit_margin": 0.0,
                   "stability": {"chance_baseline": {"pearson": 0.5}}}
        metrics.update({f"{name}_tuned": value for name, value in scores(run_logits, truth, 0.0, with_auc=False).items()})
        records = [{"class": name.split("/")[0], "image": name.split("/")[1], "predicted": "NORMAL",
                    "passes": [{"family": family, "flipped": False,
                                "pearson": float("nan") if family == "out" and index in empty else rng.uniform(0.5, 1)}
                               for family in ("in", "in", "out", "out")]}
                   for index, name in enumerate(names)]
        run = {"logits": run_logits, "labels": truth, "images": names, "metrics": metrics, "records": records,
               # The same images serve as the validation split: the checks below need its shape, not new data.
               "val_logits": run_logits, "val_labels": truth, "val_images": names}
        run.update(operating_point(run))
        return run

    runs = {("weak", 42): synthetic_run(1.0), ("strong", 42): synthetic_run(4.0, empty=(0,))}
    draws = test_draws(names, truth, samples=300)
    cache = {key: bootstrap_rows(run, draws) for key, run in runs.items()}
    arrays = {key: stability_arrays(run, np.arange(60)) for key, run in runs.items()}
    weights = subset_weights(names, samples=300)
    counts = np.array([np.bincount(index, minlength=60) for index in draws])
    together = (np.array_equal(counts[:, 30::2], counts[:, 31::2])
                and np.array_equal(weights[:, 30::2], weights[:, 31::2]))
    print(f"  a patient's images are always drawn together: {together}")
    itself = run_differences(("weak", 42), ("weak", 42), runs, cache, arrays, weights)
    zero = all(value[0] == 0 and (value[1] is None or not np.any(value[1])) for value in itself.values())
    print(f"  a run against itself: every difference 0 and every interval [0, 0]: {zero}")
    better = run_differences(("weak", 42), ("strong", 42), runs, cache, arrays, weights)
    print(f"  a better-ranking run: ROC-AUC interval above 0: {interval(better['roc_auc', ''][1])[0] > 0}")
    kept = {variant: better["stability_out_of_family", variant][2] for variant in VARIANTS}
    print(f"  an image with an empty map leaves both sides of a pair ({kept['all']} of 60 kept), and stays "
          f"in with 0 when empty maps count as 0 ({kept['empty_zero']} kept, value "
          f"{arrays['strong', 42]['out_of_family_zero'][0]}): {kept['all'] == 59 and kept['empty_zero'] == 60}")
    spread = np.linspace(-0.01, 0.01, 101)
    agree = views({"a": (0.02, 0.02 + spread, 60), "b": (0.03, 0.03 + spread, 60)})[-1]
    opposite = views({"a": (0.02, 0.02 + spread, 60), "b": (-0.03, -0.03 + spread, 60)})[-1]
    crossing = views({"a": (0.02, 0.02 + spread, 60), "b": (0.005, 0.005 + spread, 60)})[-1]
    print(f"  seed views: same sign and clear of 0 -> consistent and strict: {agree['consistent'] and agree['strict']}; "
          f"opposite signs -> neither: {not opposite['consistent'] and not opposite['strict']}; an interval "
          f"crossing 0 -> consistent, not strict: {crossing['consistent'] and not crossing['strict']}")
    print(f"  chance adjustment: baseline -> 0, 1 -> 1, halfway -> 0.5: "
          f"{adjusted(0.5, 0.5) == 0 and adjusted(1.0, 0.3) == 1 and adjusted(0.75, 0.5) == 0.5}")

    # The follow-up (2026-09-28): the architecture kinds, and the NORMAL images' margins.
    same = interaction(itself, itself)
    print(f"  an effect against itself gives an interaction of 0 everywhere: "
          f"{all(value[0] == 0 and (value[1] is None or not np.any(value[1])) for value in same.values())}")
    levels, interactions = list(architecture_pairs()), list(interaction_pairs())
    print(f"  comparisons: {len(list(pairs()))} regime and size, {len(levels)} architecture levels, "
          f"{len(interactions)} interactions (45, 27, 36): "
          f"{(len(list(pairs())), len(levels), len(interactions)) == (45, 27, 36)}")

    def cell(run_id):
        """Return an experiment's data size and regime - what a comparison of architectures must hold fixed."""
        return EXPERIMENTS[run_id].split, EXPERIMENTS[run_id].augmentation

    level_ok = all(cell(first) == cell(second) and EXPERIMENTS[first].model != EXPERIMENTS[second].model
                   for _, _, first, second in levels)
    interaction_ok = all(cell(small[0]) == cell(large[0]) and cell(small[1]) == cell(large[1])
                         and EXPERIMENTS[small[0]].model == EXPERIMENTS[small[1]].model
                         != EXPERIMENTS[large[0]].model == EXPERIMENTS[large[1]].model
                         for _, _, small, large in interactions)
    print(f"  a level compares two architectures in one cell: {level_ok}; an interaction compares the same "
          f"effect in two architectures: {interaction_ok}")
    # Five NORMAL images with margins -3 .. 1, one PNEUMONIA image far away.
    known = torch.tensor([[0.0, -3.0], [0.0, -2.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0], [0.0, 5.0]])
    point = operating_point({"logits": known, "labels": torch.tensor([0, 0, 0, 0, 0, 1]),
                             "val_logits": known, "val_labels": torch.tensor([0, 0, 0, 0, 0, 1])})
    print(f"  NORMAL margins -3..1: median -1, 90th percentile 0.6: "
          f"{point['normal_margin_median_test'] == -1 and abs(point['normal_margin_p90_test'] - 0.6) < 1e-12}")

    # The regime by data size (2026-09-28, NOTES.md #35).
    by_size = list(regime_size_pairs())
    size_ok = all(len({EXPERIMENTS[run_id].model for run_id in small + large}) == 1
                  and cell(small[0])[0] == cell(small[1])[0] == "train_20" != cell(large[0])[0] == cell(large[1])[0]
                  and cell(small[0])[1] == cell(large[0])[1] == "none" != cell(small[1])[1] == cell(large[1])[1]
                  for _, _, small, large in by_size)
    print(f"  regime by data size: {len(by_size)} comparisons (12), each one architecture and one regime pair, "
          f"at 20% and at a larger size: {len(by_size) == 12 and size_ok}")
    # Hand-made rows: the new row equals the augmented minus the unaugmented change, then one that does not.
    shared = {"model": "M", "data_size": "", "metric": "roc_auc", "variant": "", "view": "seed 42"}
    made = [dict(shared, kind="data size", contrast="100%-20%", augmentation=LOG_LABELS["strong"], difference=0.05),
            dict(shared, kind="data size", contrast="100%-20%", augmentation=LOG_LABELS["none"], difference=0.02),
            dict(shared, kind="regime by data size", contrast="strong-none", data_size="100%-20%", augmentation="",
                 difference=0.03)]
    wrong = made[:2] + [dict(made[2], difference=0.04)]
    print(f"  the two ways agree on a right row and catch a wrong one: "
          f"{two_ways(made) < EXACT and abs(two_ways(wrong) - 0.01) < EXACT}")


def main():
    """Collect on Colab, run the self-check, or - with no option - run the analysis."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collect", action="store_true",
                        help="re-score every experiment and gather its files under results/runs/ (GPU)")
    parser.add_argument("--seed", type=int, default=SEED, help=f"which sweep to collect (default {SEED})")
    parser.add_argument("--id", action="append", help="collect only this experiment; repeatable")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (default: 0)")
    parser.add_argument("--output-root", type=Path, default=None, help="where checkpoints/ and results/ live")
    parser.add_argument("--check", action="store_true", help="self-check on synthetic data")
    args = parser.parse_args()

    if args.collect:
        verdicts = list(collect(args.seed, args.output_root, args.workers, args.id).values())
        print(f"seed {args.seed}: " + ", ".join(f"{verdicts.count(kind)} {kind}"
                                                 for kind in ("PASS-exact", "PASS-approx", "FAIL")))
    elif args.check:
        self_check()
    else:
        analyse(args.output_root)


if __name__ == "__main__":
    main()
