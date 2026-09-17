"""Classification metrics for a trained model, shared by validation during training and the test run.

train.py scores the validation set with these functions every epoch, and the final test
evaluation uses the same ones, so validation and test numbers can never be computed
differently. PNEUMONIA (label 1) is the positive class.

The headline numbers for the 27 experiments carry a 95% confidence interval from a percentile
bootstrap that resamples **patient groups**, not individual images (NOTES.md, the
confidence-interval decision): one patient contributes several test images, and resampling
images as if they were independent would make every interval too narrow. The interval covers
test-sampling noise only - how much a metric would move on another test set of this size. It
says nothing about training noise, which is question #21's seed-to-seed comparison.

    from src.evaluate import predict, scores, write_metrics
    logits, labels = predict(model, test_loader, device)
    metrics = write_metrics(config, logits, labels, test_loader.dataset.paths)

or run it to self-check the metrics, the bootstrap and the figures on synthetic data:

    python -m src.evaluate
"""

import json

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import LOG_LABELS
from src.paths import output_root, patient_key

BOOTSTRAP_SAMPLES = 2000
CONFIDENCE = 95


def predict(model, loader, device):
    """Run the model over every batch of a loader and return (logits, labels) as CPU tensors."""
    model.eval()
    all_logits, all_labels = [], []
    # no_grad skips recording operations for backpropagation - less memory, and nothing here trains.
    with torch.no_grad():
        for images, labels in loader:
            all_logits.append(model(images.to(device)).cpu())
            all_labels.append(labels)
    # torch.cat joins the per-batch tensors end to end into one tensor per quantity.
    return torch.cat(all_logits), torch.cat(all_labels)


def scores(logits, labels):
    """Return every classification metric for two-logit outputs, as a dict of floats.

    ROC-AUC uses the PNEUMONIA probability; the rest use the predicted class, which for two
    logits is the larger one - the same as a 0.5 threshold on the probability. recall is on
    PNEUMONIA, specificity the same quantity on NORMAL, so the report can name both error rates.
    """
    # softmax turns each row of logits into probabilities summing to 1; column 1 is PNEUMONIA.
    probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
    predictions = logits.argmax(dim=1).numpy()
    truth = labels.numpy()
    # zero_division=0 keeps a degenerate input (no predicted positives, say) from raising; it can
    # happen in a bootstrap resample, never in a full evaluation.
    return {
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "specificity": float(recall_score(truth, predictions, pos_label=0, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
    }


def confusion_counts(logits, labels):
    """Return the confusion matrix as {tn, fp, fn, tp}."""
    # labels=[0, 1] forces a 2x2 matrix even when one class is missing from the input; without it
    # a single-class input returns a 1x1 matrix and the four names below would be misassigned.
    matrix = confusion_matrix(labels.numpy(), logits.argmax(dim=1).numpy(), labels=[0, 1])
    # .ravel() flattens the 2x2 matrix to [tn, fp, fn, tp] in reading order.
    tn, fp, fn, tp = matrix.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def patient_groups(paths):
    """Return one array of image positions per patient group, for the bootstrap to resample.

    Images whose filename matches no pattern become their own single-image group, which is the
    conservative reading: nothing is merged on the strength of a failed parse.
    """
    groups = {}
    for position, path in enumerate(paths):
        # The class is the parent folder name, the same rule ChestXrayDataset uses for labels.
        key = patient_key(path.parent.name, path) or f"unmatched:{path.name}"
        groups.setdefault(key, []).append(position)
    return [np.array(positions) for positions in groups.values()]


def bootstrap_intervals(logits, labels, groups, samples=BOOTSTRAP_SAMPLES, seed=0):
    """Return ({metric}_ci_low / _ci_high for every metric, number of resamples skipped).

    Percentile bootstrap: resample the groups with replacement, take all images of the groups
    drawn, recompute every metric, and report the 2.5th and 97.5th percentiles. Percentile
    intervals are slightly biased for metrics close to 0 or 1, which is the accepted cost of
    using one mechanism for all of them. A resample that draws only one class cannot produce a
    ROC-AUC, so it is skipped and counted.
    """
    # default_rng is NumPy's current generator; seeding it makes the interval reproducible.
    rng = np.random.default_rng(seed)
    collected = {}
    skipped = 0
    for _ in range(samples):
        # One draw of len(groups) group positions, with replacement.
        drawn = rng.integers(0, len(groups), len(groups))
        # concatenate glues the chosen groups' image positions into one index array.
        index = np.concatenate([groups[position] for position in drawn])
        sample_labels = labels[index]
        # .unique() on the labels tells us whether both classes survived the draw.
        if len(torch.unique(sample_labels)) < 2:
            skipped += 1
            continue
        for name, value in scores(logits[index], sample_labels).items():
            collected.setdefault(name, []).append(value)

    tail = (100 - CONFIDENCE) / 2
    intervals = {}
    for name, values in collected.items():
        low, high = np.percentile(values, [tail, 100 - tail])
        intervals[f"{name}_ci_low"] = float(low)
        intervals[f"{name}_ci_high"] = float(high)
    return intervals, skipped


def metrics_payload(config, logits, labels, paths, samples=BOOTSTRAP_SAMPLES):
    """Build the full results dict for one run, in the schema NOTES.md records for question #13."""
    groups = patient_groups(paths)
    intervals, skipped = bootstrap_intervals(logits, labels, groups, samples, seed=config.seed)
    return {
        "experiment_id": config.run_id,
        # The display names keep this file readable next to experiments/experiment_log.md.
        "model": LOG_LABELS[config.model],
        "data_size": LOG_LABELS[config.split],
        "augmentation": LOG_LABELS[config.augmentation],
        "seed": config.seed,
        "fine_tuning": config.fine_tuning,
        "weighted_loss": config.weighted_loss,
        "images": len(labels),
        "patient_groups": len(groups),
        **scores(logits, labels),
        **intervals,
        "confusion_matrix": confusion_counts(logits, labels),
        # Filled in by src/stability.py (workplan section 7); null until then, so the schema is
        # complete from the first run and Or's results table has the columns it expects.
        "gradcam_stability_in_family": None,
        "gradcam_stability_out_of_family": None,
        "bootstrap": {"samples": samples, "skipped": skipped, "seed": config.seed,
                      "unit": "patient group", "confidence": CONFIDENCE},
    }


def results_directory(root=None, kind="metrics"):
    """Return <output root>/results/<kind>, creating it if needed."""
    directory = (root or output_root()) / "results" / kind
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_metrics(config, logits, labels, paths, root=None, samples=BOOTSTRAP_SAMPLES):
    """Write results/metrics/<ID>_metrics.json for one run and return the payload."""
    payload = metrics_payload(config, logits, labels, paths, samples)
    path = results_directory(root) / f"{config.run_id}_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_metrics(root=None):
    """Read every results/metrics/*_metrics.json under the output root, ordered by run ID."""
    directory = results_directory(root)
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*_metrics.json"))]


def comparison_table(rows):
    """Return the combined results table as GitHub-flavoured markdown, one line per experiment.

    The table the report and experiments/experiment_log.md are filled from. ROC-AUC carries its
    interval, because a gap smaller than the interval is not a result.
    """
    header = ("| ID | Model | Data | Augmentation | ROC-AUC (95% CI) | F1 | Balanced acc. | "
              "Recall | Specificity | Stability (out) |")
    lines = [header, "|" + "---|" * 10]
    for row in rows:
        interval = ""
        if row.get("roc_auc_ci_low") is not None:
            interval = f" ({row['roc_auc_ci_low']:.3f}-{row['roc_auc_ci_high']:.3f})"
        stability = row.get("gradcam_stability_out_of_family")
        lines.append(
            f"| {row['experiment_id']} | {row['model']} | {row['data_size']} | {row['augmentation']} "
            f"| {row['roc_auc']:.3f}{interval} | {row['f1']:.3f} | {row['balanced_accuracy']:.3f} "
            f"| {row['recall']:.3f} | {row['specificity']:.3f} "
            f"| {'-' if stability is None else f'{stability:.3f}'} |")
    return '\n'.join(lines)


def main():
    """Self-check every function on synthetic data, without needing a trained model."""
    rng = np.random.default_rng(0)
    # 600 images over 200 "patients" of 3 images each, a believable class ratio, and a model
    # that is good but not perfect.
    labels = torch.tensor(rng.choice([0, 1], size=600, p=[0.375, 0.625]))
    margin = torch.tensor(rng.normal(labels.numpy() * 2.0 - 1.0, 1.2), dtype=torch.float32)
    logits = torch.stack([torch.zeros(600), margin], dim=1)

    values = scores(logits, labels)
    print("  metrics:", {name: round(value, 4) for name, value in values.items()})
    print(f"  every metric in [0, 1]: {all(0 <= value <= 1 for value in values.values())}")

    counts = confusion_counts(logits, labels)
    print(f"  confusion {counts}, sums to {sum(counts.values())} of {len(labels)}")
    single = confusion_counts(logits[labels == 1], labels[labels == 1])
    print(f"  single-class input still 2x2: {sorted(single) == ['fn', 'fp', 'tn', 'tp']}")

    # The same data grouped two ways: 3 images per patient, against every image its own patient.
    grouped = [np.arange(position, position + 3) for position in range(0, 600, 3)]
    singles = [np.array([position]) for position in range(600)]
    wide, _ = bootstrap_intervals(logits, labels, grouped, samples=400, seed=1)
    narrow, skipped = bootstrap_intervals(logits, labels, singles, samples=400, seed=1)
    width = lambda intervals, name: intervals[f"{name}_ci_high"] - intervals[f"{name}_ci_low"]
    print(f"  roc_auc 95% CI by patient {width(wide, 'roc_auc'):.4f} wide, by image "
          f"{width(narrow, 'roc_auc'):.4f} - grouping widens it: "
          f"{width(wide, 'roc_auc') > width(narrow, 'roc_auc')}")
    print(f"  intervals contain the point estimate: "
          f"{all(wide[f'{n}_ci_low'] <= v <= wide[f'{n}_ci_high'] for n, v in values.items())}, "
          f"resamples skipped: {skipped}")


if __name__ == "__main__":
    main()
