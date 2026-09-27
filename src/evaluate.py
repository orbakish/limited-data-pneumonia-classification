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

from src.config import LOG_LABELS, output_stem
from src.paths import output_root, patient_key

BOOTSTRAP_SAMPLES = 2000
CONFIDENCE = 95
# Recorded with every tuned threshold, so a results file says how its threshold was chosen.
THRESHOLD_RULE = "max validation balanced accuracy on the selected checkpoint"


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


def margins(logits):
    """Return each image's logit margin, logit_PNEUMONIA - logit_NORMAL, as a float64 NumPy array.

    A two-logit model predicts PNEUMONIA exactly when this margin is above 0, and a decision
    threshold is a different cut-off on it (NOTES.md #27). The margin is used rather than the
    softmax probability because a confident model's float32 probability rounds to exactly 1.0,
    so many images would tie; their margins stay distinct. float64 keeps the midpoints that
    choose_threshold places between neighbouring margins from rounding onto a margin.
    """
    # .double() converts to float64 before subtracting, so the difference is exact.
    return (logits[:, 1].double() - logits[:, 0].double()).numpy()


def classify(logits, threshold=0.0):
    """Return the predicted class of every image: 1 (PNEUMONIA) where the margin exceeds threshold.

    At the default threshold of 0 this is exactly the argmax of the two logits, including an
    exact tie, which both send to NORMAL.
    """
    # The comparison gives a True/False array; .astype(np.int64) turns it into 1/0 class labels.
    return (margins(logits) > threshold).astype(np.int64)


def scores(logits, labels, threshold=0.0, with_auc=True):
    """Return every classification metric for two-logit outputs, as a dict of floats.

    ROC-AUC uses the PNEUMONIA probability and ignores the threshold; the rest use the class
    from classify(), so the default threshold of 0 is the plain argmax - the same as 0.5 on the
    probability. recall is on PNEUMONIA, specificity the same quantity on NORMAL, so the report
    can name both error rates. with_auc=False skips ROC-AUC, for callers that only need the
    threshold metrics.
    """
    predictions = classify(logits, threshold)
    truth = labels.numpy()
    values = {}
    if with_auc:
        # softmax turns each row of logits into probabilities summing to 1; column 1 is PNEUMONIA.
        # Kept in float32 exactly as before, so every ROC-AUC already recorded stays reproducible.
        probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
        values["roc_auc"] = float(roc_auc_score(truth, probabilities))
    # zero_division=0 keeps a degenerate input (no predicted positives, say) from raising; it can
    # happen in a bootstrap resample, never in a full evaluation.
    values.update({
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "specificity": float(recall_score(truth, predictions, pos_label=0, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
    })
    return values


def confusion_counts(logits, labels, threshold=0.0):
    """Return the confusion matrix at the given margin threshold as {tn, fp, fn, tp}."""
    # labels=[0, 1] forces a 2x2 matrix even when one class is missing from the input; without it
    # a single-class input returns a 1x1 matrix and the four names below would be misassigned.
    matrix = confusion_matrix(labels.numpy(), classify(logits, threshold), labels=[0, 1])
    # .ravel() flattens the 2x2 matrix to [tn, fp, fn, tp] in reading order.
    tn, fp, fn, tp = matrix.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def choose_threshold(logits, labels):
    """Return the margin threshold that maximises balanced accuracy on these images, as a float.

    Run on the validation set with the run's selected checkpoint; the test set then uses the
    result unchanged (NOTES.md #27). Maximising balanced accuracy is maximising Youden's J,
    because J = sensitivity + specificity - 1 = 2 x balanced accuracy - 1.

    Candidates are 0 - the plain argmax - and the midpoints between neighbouring distinct
    margins: every threshold inside one gap gives the same predictions, so one per gap is
    enough, and a midpoint is never equal to a margin. Ties go to the candidate closest to 0,
    which keeps the default whenever it is already optimal, then to the smaller one.
    """
    values = margins(logits)
    truth = labels.numpy()
    if len(np.unique(truth)) < 2:
        raise ValueError("choosing a threshold needs images of both classes")
    # np.unique also sorts, so neighbouring entries are neighbouring margins.
    distinct = np.unique(values)
    candidates = np.concatenate([[0.0], (distinct[:-1] + distinct[1:]) / 2])
    # Broadcasting compares every candidate (rows) with every image (columns) in one step.
    predicted = values[None, :] > candidates[:, None]
    # Counts, not means, so two candidates with the same counts get exactly the same score.
    sensitivity = predicted[:, truth == 1].sum(axis=1) / (truth == 1).sum()
    specificity = (~predicted[:, truth == 0]).sum(axis=1) / (truth == 0).sum()
    balanced = (sensitivity + specificity) / 2
    winners = candidates[balanced == balanced.max()]
    # lexsort sorts by its last key first: distance from 0, then the value itself.
    return float(winners[np.lexsort((winners, np.abs(winners)))[0]])


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


def bootstrap_intervals(logits, labels, groups, samples=BOOTSTRAP_SAMPLES, seed=0, threshold=0.0,
                        with_auc=True):
    """Return ({metric}_ci_low / _ci_high for every metric, number of resamples skipped).

    Percentile bootstrap: resample the groups with replacement, take all images of the groups
    drawn, recompute every metric, and report the 2.5th and 97.5th percentiles. Percentile
    intervals are slightly biased for metrics close to 0 or 1, which is the accepted cost of
    using one mechanism for all of them. A resample that draws only one class cannot produce a
    ROC-AUC, so it is skipped and counted.

    The threshold stays fixed in every resample: a tuned threshold comes from validation, so
    the interval covers test-sampling noise given that threshold, not the noise of choosing it.
    The same seed draws the same resamples, whatever the threshold.
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
        for name, value in scores(logits[index], sample_labels, threshold, with_auc).items():
            collected.setdefault(name, []).append(value)

    tail = (100 - CONFIDENCE) / 2
    intervals = {}
    for name, values in collected.items():
        low, high = np.percentile(values, [tail, 100 - tail])
        intervals[f"{name}_ci_low"] = float(low)
        intervals[f"{name}_ci_high"] = float(high)
    return intervals, skipped


def threshold_fields(threshold):
    """Return the fields that record a tuned threshold: the margin, its probability, and the rule."""
    return {
        # The margin is the value actually applied; the probability is for reading, and for a
        # very confident model it can round to exactly 1.0.
        "threshold_logit_margin": threshold,
        # sigmoid turns a margin into the PNEUMONIA probability it corresponds to.
        "threshold": float(torch.sigmoid(torch.tensor(threshold, dtype=torch.float64))),
        "threshold_rule": THRESHOLD_RULE,
    }


def tuned_fields(logits, labels, groups, threshold, samples, seed):
    """Return the threshold and every threshold metric at it, with intervals, all keyed *_tuned.

    ROC-AUC has no tuned version: it does not depend on a threshold.
    """
    fields = threshold_fields(threshold)
    fields.update({f"{name}_tuned": value
                   for name, value in scores(logits, labels, threshold, with_auc=False).items()})
    intervals, _ = bootstrap_intervals(logits, labels, groups, samples, seed, threshold, with_auc=False)
    # balanced_accuracy_ci_low becomes balanced_accuracy_tuned_ci_low, and so on.
    fields.update({key.replace("_ci_", "_tuned_ci_"): value for key, value in intervals.items()})
    fields["confusion_matrix_tuned"] = confusion_counts(logits, labels, threshold)
    return fields


def metrics_payload(config, logits, labels, paths, samples=BOOTSTRAP_SAMPLES, threshold=None, stability=None):
    """Build the full results dict for one run, in the schema NOTES.md records for question #13.

    Every unsuffixed metric is at the default threshold (0.5). Given a tuned threshold, the
    payload also carries the same metrics at that threshold, keyed *_tuned (NOTES.md #27).
    stability is the dict src/stability.py returns for an experiment; without it the two
    stability fields stay null.
    """
    groups = patient_groups(paths)
    intervals, skipped = bootstrap_intervals(logits, labels, groups, samples, seed=config.seed)
    payload = {
        "experiment_id": config.run_id,
        # The display names keep this file readable next to experiments/experiment_log.md.
        "model": LOG_LABELS[config.model],
        "data_size": LOG_LABELS[config.split],
        "augmentation": LOG_LABELS[config.augmentation],
        "seed": config.seed,
        "fine_tuning": config.fine_tuning,
        "weighted_loss": config.weighted_loss,
        # ** unpacks a dict into this one. The key appears only for the T16 diagnostic, so the
        # E-run schema agreed with Or (NOTES.md #13) stays unchanged.
        **({"bn_momentum": config.bn_momentum} if config.bn_momentum is not None else {}),
        "images": len(labels),
        "patient_groups": len(groups),
        **scores(logits, labels),
        **intervals,
        "confusion_matrix": confusion_counts(logits, labels),
        # Filled in by src/stability.py for an experiment (NOTES.md #7); null otherwise, so the
        # schema is complete in every file and Or's results table has the columns it expects.
        **(stability or {"gradcam_stability_in_family": None, "gradcam_stability_out_of_family": None}),
        "bootstrap": {"samples": samples, "skipped": skipped, "seed": config.seed,
                      "unit": "patient group", "confidence": CONFIDENCE},
    }
    if threshold is not None:
        payload.update(tuned_fields(logits, labels, groups, threshold, samples, config.seed))
    return payload


def results_directory(root=None, kind="metrics"):
    """Return <output root>/results/<kind>, creating it if needed."""
    directory = (root or output_root()) / "results" / kind
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_metrics(payload, config, root=None):
    """Write a results payload to results/metrics/<ID>_metrics.json and return the path.

    run_all.py counts a run as finished once this file exists, so it must never exist half
    written: the JSON goes to a temporary file first, and Path.replace renames it in one step.
    """
    path = results_directory(root) / f"{output_stem(config)}_metrics.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def write_metrics(config, logits, labels, paths, root=None, samples=BOOTSTRAP_SAMPLES, threshold=None,
                  stability=None):
    """Build and save results/metrics/<ID>_metrics.json for one run and return the payload."""
    payload = metrics_payload(config, logits, labels, paths, samples, threshold, stability)
    save_metrics(payload, config, root)
    return payload


def write_validation_metrics(config, logits, labels, paths, run_dir, samples=BOOTSTRAP_SAMPLES,
                             threshold=None):
    """Write validation_metrics.json beside a run's checkpoints and return the payload.

    A tuning run never touches the test set, so its numbers have to come from validation - and
    experiments/tuning_log.md asks for a confidence interval, which only the bootstrap provides.
    An experiment writes one too, recording the threshold chosen here; it gets no tuned
    validation metrics, which would be scored on the images the threshold was chosen on. Kept in
    the run folder rather than results/metrics/, which is for test results.
    """
    payload = metrics_payload(config, logits, labels, paths, samples)
    payload["split"] = "val"
    if threshold is not None:
        payload.update(threshold_fields(threshold))
    (run_dir / "validation_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_metrics(root=None):
    """Read every results/metrics/*_metrics.json under the output root, ordered by run ID."""
    directory = results_directory(root)
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*_metrics.json"))]


def comparison_table(rows):
    """Return the combined results table as GitHub-flavoured markdown, one line per experiment.

    The table the report and experiments/experiment_log.md are filled from. ROC-AUC carries its
    interval, because a gap smaller than the interval is not a result. The threshold metrics are
    at the validation-chosen threshold (NOTES.md #27), with balanced accuracy at 0.5 beside them.
    """
    def cell(value):
        """Format one number, or a dash for a field this row does not have."""
        return "-" if value is None else f"{value:.3f}"

    header = ("| ID | Model | Data | Augmentation | ROC-AUC (95% CI) | Threshold | Balanced acc. | F1 | "
              "Recall | Specificity | Balanced acc. @0.5 | Stability (in) | Stability (out) | Gap |")
    lines = [header, "|" + "---|" * 14]
    for row in rows:
        interval = ""
        if row.get("roc_auc_ci_low") is not None:
            interval = f" ({row['roc_auc_ci_low']:.3f}-{row['roc_auc_ci_high']:.3f})"
        lines.append(
            f"| {row['experiment_id']} | {row['model']} | {row['data_size']} | {row['augmentation']} "
            f"| {row['roc_auc']:.3f}{interval} | {cell(row.get('threshold'))} "
            f"| {cell(row.get('balanced_accuracy_tuned'))} | {cell(row.get('f1_tuned'))} "
            f"| {cell(row.get('recall_tuned'))} | {cell(row.get('specificity_tuned'))} "
            f"| {cell(row['balanced_accuracy'])} | {cell(row.get('gradcam_stability_in_family'))} "
            f"| {cell(row.get('gradcam_stability_out_of_family'))} | {cell(row.get('gradcam_stability_gap'))} |")
    return "\n".join(lines)


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

    # The default threshold must reproduce the old argmax and softmax path exactly, or the
    # per-epoch history and every tuning run already recorded would change (NOTES.md #27).
    truth, argmax = labels.numpy(), logits.argmax(dim=1).numpy()
    old = {"roc_auc": float(roc_auc_score(truth, torch.softmax(logits, dim=1)[:, 1].numpy())),
           "balanced_accuracy": float(balanced_accuracy_score(truth, argmax)),
           "accuracy": float(accuracy_score(truth, argmax)),
           "precision": float(precision_score(truth, argmax, zero_division=0)),
           "recall": float(recall_score(truth, argmax, zero_division=0)),
           "specificity": float(recall_score(truth, argmax, pos_label=0, zero_division=0)),
           "f1": float(f1_score(truth, argmax, zero_division=0))}
    tie = torch.tensor([[1.0, 1.0]])
    print(f"  threshold 0 equals the old argmax path exactly: {scores(logits, labels) == old}, "
          f"exact tie goes to NORMAL in both: {classify(tie)[0] == tie.argmax(dim=1)[0] == 0}")

    # A model biased towards PNEUMONIA: the chosen threshold should move up and help.
    shifted = logits + torch.tensor([0.0, 1.5])
    chosen = choose_threshold(shifted, labels)
    before = scores(shifted, labels)["balanced_accuracy"]
    after = scores(shifted, labels, chosen)["balanced_accuracy"]
    print(f"  shifted model: threshold {chosen:+.3f}, balanced accuracy {before:.4f} -> {after:.4f}")

    # Constructed ties, where the answer is known in advance.
    def pick(values, classes):
        """Return choose_threshold for margins given directly (NORMAL logit fixed at 0)."""
        margin = torch.tensor(values, dtype=torch.float32)
        return choose_threshold(torch.stack([torch.zeros(len(values)), margin], dim=1),
                                torch.tensor(classes))
    print(f"  optimal gap contains 0 -> 0: {pick([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1]) == 0.0}; "
          f"tied gaps at 0.75 and 2.5 -> 0.75: {pick([0.5, 1.0, 2.0, 3.0], [0, 1, 0, 1]) == 0.75}; "
          f"tied at -0.75 and 0.75 -> -0.75: {pick([-1.0, -0.5, 0.5, 1.0], [0, 1, 0, 1]) == -0.75}; "
          f"all margins equal -> 0: {pick([0.3, 0.3, 0.3], [0, 1, 1]) == 0.0}")

    tuned = tuned_fields(shifted, labels, grouped, chosen, 400, seed=1)
    print(f"  tuned intervals contain their estimates: "
          f"{all(tuned[f'{n}_tuned_ci_low'] <= tuned[f'{n}_tuned'] <= tuned[f'{n}_tuned_ci_high'] for n in old if n != 'roc_auc')}, "
          f"no tuned ROC-AUC: {'roc_auc_tuned' not in tuned}")


if __name__ == "__main__":
    main()
