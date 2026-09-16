"""Every figure the project produces, and the audit that checks they are labelled correctly.

Two kinds live here so that one palette, one title format and one save path serve both:

* Per run, written by run_experiment.py: the training curves and the confusion matrix
  (E##_training_curves.png, E##_confusion_matrix.png; Grad-CAM examples come from
  src/gradcam.py in workplan section 6).
* Across runs, for the report: one panel per architecture, a line per augmentation regime,
  against training-set size - with confidence intervals - plus the in-family / out-of-family
  stability gap.

Colour always means the augmentation regime, in the fixed order none / mild / strong, and each
regime also has its own marker, so the figures survive greyscale printing and colour-blind
readers. The palette was checked with the six-check validator: colour-blind separation 11.0 and
normal-vision separation 25.8 (both above their floors), and every colour clears 3:1 contrast
against white paper.

Every figure stores its title inside the PNG, so `python -m src.figures --audit` can confirm
that all 81 report figures carry the label their config says they should, instead of anyone
checking by eye.

    from src.figures import confusion_figure, curves_figure
    python -m src.figures            # self-check on synthetic results
    python -m src.figures --audit    # verify the figures under the output root
"""

import argparse
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from src.config import (AUGMENTATIONS, EXPERIMENTS, LOG_LABELS, MODELS, RECIPE, TRAIN_SPLITS,
                        TUNING_RUNS, output_stem, run_title)
from src.paths import output_root

# One hue per augmentation regime, in this order, everywhere in the report. Validated with the
# six checks (see the module docstring); do not swap in an untested colour.
REGIME_COLOURS = {"none": "#0072B2", "mild": "#D55E00", "strong": "#009E73"}
# A second, redundant encoding: identity never rests on colour alone.
REGIME_MARKERS = {"none": "o", "mild": "s", "strong": "^"}
SOFTWARE = "limited-data-pneumonia-classification"


def figures_directory(root=None):
    """Return <output root>/results/figures, creating it if needed."""
    directory = (root or output_root()) / "results" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def titled(figure, title, size=9):
    """Put the title on the figure, wrapped, so a long run label is never clipped."""
    # A run title plus the figure kind runs past 70 characters, which a 4-inch figure cannot
    # show on one line; fill() breaks it into lines instead of letting it overflow the canvas.
    figure.suptitle(textwrap.fill(title, 72), fontsize=size)


def save_figure(figure, path, title):
    """Save a figure as PNG with its title stored inside the file, and return the path.

    The embedded title is what --audit reads back: a figure whose label does not match the run
    it claims to describe is a mislabelled result, and this makes that checkable by command.
    """
    # metadata= writes PNG text chunks, the same place an image viewer shows "Title".
    figure.savefig(path, dpi=150, metadata={"Title": title, "Software": SOFTWARE})
    return path


def confusion_figure(counts, title, path):
    """Draw the confusion matrix as a labelled 2x2 grid of counts."""
    grid = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    figure = Figure(figsize=(5.0, 4.6), layout="constrained")
    axes = figure.add_subplot()
    # A single hue, light to dark: the cell value is a magnitude, not a category.
    axes.imshow(grid, cmap="Blues")
    axes.set_xticks([0, 1], ["predicted\nNORMAL", "predicted\nPNEUMONIA"], fontsize=8)
    axes.set_yticks([0, 1], ["true\nNORMAL", "true\nPNEUMONIA"], fontsize=8)
    for row in (0, 1):
        for column in (0, 1):
            count = grid[row, column]
            # Dark cells need light text and light cells dark text, or half the numbers vanish.
            colour = "white" if count > grid.max() / 2 else "#1a1a19"
            axes.text(column, row, f"{count}", ha="center", va="center", color=colour, fontsize=15)
    # The two error types are what the report argues about, so name them on the figure itself.
    axes.set_xlabel("top right: false alarms      bottom left: missed pneumonia", fontsize=7)
    titled(figure, title, size=8)
    return save_figure(figure, path, title)


def curves_figure(history, title, path):
    """Plot loss and the two validation metrics per epoch, marking the checkpointed epoch."""
    epochs = [row["epoch"] for row in history]
    figure = Figure(figsize=(8.4, 3.4), layout="constrained")
    loss_axes, metric_axes = figure.subplots(1, 2)

    loss_axes.plot(epochs, [row["train_loss"] for row in history], color="#0072B2", label="train")
    loss_axes.plot(epochs, [row["val_loss"] for row in history], color="#D55E00", label="validation")
    loss_axes.set_xlabel("epoch")
    loss_axes.set_ylabel("cross-entropy loss")
    loss_axes.legend(fontsize=8, frameon=False)

    for name, colour in (("roc_auc", "#0072B2"), ("balanced_accuracy", "#009E73")):
        # A metric missing from an older history file should not stop the figure being drawn.
        if name in history[0]:
            metric_axes.plot(epochs, [row[name] for row in history], color=colour,
                             label=name.replace("_", " "))
    # Mark the epoch the checkpoint actually holds - the best on the selection metric, which
    # since 2026-09-16 is balanced accuracy rather than ROC-AUC.
    best = max(history, key=lambda row: row[RECIPE.selection_metric])["epoch"]
    metric_axes.axvline(best, linestyle="--", linewidth=1, color="#6b6b68")
    metric_axes.set_xlabel(f"epoch (checkpoint kept: {best})")
    metric_axes.set_ylabel("validation metric")
    metric_axes.legend(fontsize=8, frameon=False)

    for axes in (loss_axes, metric_axes):
        # Recessive grid: it helps read values without competing with the lines.
        axes.grid(alpha=0.25, linewidth=0.6)
        axes.set_axisbelow(True)
    titled(figure, title)
    return save_figure(figure, path, title)


def by_model_and_regime(rows, metric):
    """Return {model: {regime: ([sizes], [values], [ci_low], [ci_high])}} from metrics dicts."""
    collected = {}
    for model in MODELS:
        for regime in AUGMENTATIONS:
            sizes, values, lows, highs = [], [], [], []
            for split in TRAIN_SPLITS:
                match = [row for row in rows if row["model"] == LOG_LABELS[model]
                         and row["augmentation"] == LOG_LABELS[regime]
                         and row["data_size"] == LOG_LABELS[split]]
                if not match or match[0].get(metric) is None:
                    continue
                row = match[0]
                # int("20%"[:-1]) -> 20, so the x axis is a real number line, not three labels.
                sizes.append(int(row["data_size"].rstrip("%")))
                values.append(row[metric])
                lows.append(row.get(f"{metric}_ci_low", row[metric]))
                highs.append(row.get(f"{metric}_ci_high", row[metric]))
            collected.setdefault(model, {})[regime] = (sizes, values, lows, highs)
    return collected


def comparison_figure(rows, metric, title, path, ylabel=None):
    """One panel per architecture: the metric against training-set size, a line per regime.

    Small multiples rather than nine lines on one axis, and confidence intervals as vertical
    bars so a difference smaller than the noise is visible as such.
    """
    collected = by_model_and_regime(rows, metric)
    figure = Figure(figsize=(9.6, 3.9), layout="constrained")
    panels = figure.subplots(1, len(MODELS), sharey=True)
    for panel, model in zip(panels, MODELS):
        for regime in AUGMENTATIONS:
            sizes, values, lows, highs = collected[model][regime]
            if not sizes:
                continue
            # yerr wants distances from the point, not absolute bounds.
            errors = [np.array(values) - np.array(lows), np.array(highs) - np.array(values)]
            panel.errorbar(sizes, values, yerr=errors, label=regime, color=REGIME_COLOURS[regime],
                           marker=REGIME_MARKERS[regime], markersize=6, linewidth=2, capsize=3)
        panel.set_title(LOG_LABELS[model], fontsize=9)
        panel.set_xlabel("training data (%)")
        panel.set_xticks([20, 50, 100])
        panel.grid(alpha=0.25, linewidth=0.6)
        panel.set_axisbelow(True)
    panels[0].set_ylabel(ylabel or metric.replace("_", " "))
    # One legend for the whole figure, below the panels: inside a panel it would sooner or later
    # sit on top of a line. loc="outside lower center" is a constrained-layout placement.
    figure.legend(*panels[0].get_legend_handles_labels(), title="augmentation", ncols=3,
                  fontsize=8, title_fontsize=8, frameon=False, loc="outside lower center")
    titled(figure, title)
    return save_figure(figure, path, title)


def stability_gap_figure(rows, title, path):
    """Plot in-family minus out-of-family stability: how much apparent stability is trained in."""
    in_family = by_model_and_regime(rows, "gradcam_stability_in_family")
    out_family = by_model_and_regime(rows, "gradcam_stability_out_of_family")
    figure = Figure(figsize=(9.6, 3.9), layout="constrained")
    panels = figure.subplots(1, len(MODELS), sharey=True)
    for panel, model in zip(panels, MODELS):
        for regime in AUGMENTATIONS:
            sizes, inside, _, _ = in_family[model][regime]
            _, outside, _, _ = out_family[model][regime]
            if not sizes or not outside:
                continue
            gap = np.array(inside) - np.array(outside)
            panel.plot(sizes, gap, label=regime, color=REGIME_COLOURS[regime],
                       marker=REGIME_MARKERS[regime], markersize=6, linewidth=2)
        # Zero is the meaningful reference: no gap means no trained-in invariance.
        panel.axhline(0, color="#6b6b68", linewidth=1)
        panel.set_title(LOG_LABELS[model], fontsize=9)
        panel.set_xlabel("training data (%)")
        panel.set_xticks([20, 50, 100])
        panel.grid(alpha=0.25, linewidth=0.6)
        panel.set_axisbelow(True)
    panels[0].set_ylabel("in-family minus out-of-family")
    figure.legend(*panels[0].get_legend_handles_labels(), title="augmentation", ncols=3,
                  fontsize=8, title_fontsize=8, frameon=False, loc="outside lower center")
    titled(figure, title)
    return save_figure(figure, path, title)


def report_figures(rows, root=None):
    """Draw every cross-experiment figure from the metrics files and return their paths."""
    directory = figures_directory(root)
    drawn = [
        comparison_figure(rows, "roc_auc", "Test ROC-AUC by data size and augmentation regime",
                          directory / "summary_roc_auc.png", "test ROC-AUC (95% CI)"),
        comparison_figure(rows, "f1", "Test F1 by data size and augmentation regime",
                          directory / "summary_f1.png", "test F1 (95% CI)"),
    ]
    # Stability figures need src/stability.py to have filled those fields in.
    if any(row.get("gradcam_stability_out_of_family") is not None for row in rows):
        drawn.append(comparison_figure(
            rows, "gradcam_stability_out_of_family",
            "Grad-CAM stability under noise (out-of-family) by data size and regime",
            directory / "summary_stability.png", "cosine similarity"))
        drawn.append(stability_gap_figure(
            rows, "Trained-in invariance: in-family minus out-of-family stability",
            directory / "summary_stability_gap.png"))
    return drawn


def read_title(path):
    """Return the title stored inside a PNG, or None if it has none."""
    with Image.open(path) as image:
        # .text holds the PNG's text chunks; matplotlib writes the title into "Title".
        return image.text.get("Title")


def audit(root=None):
    """Check every run's figures exist and carry the title their config implies.

    Returns (rows, problems): one row per expected figure, and the subset that is missing or
    mislabelled. An experiment expects three figures; a tuning run only has training curves,
    since it never reaches the test split or Grad-CAM.
    """
    directory = figures_directory(root)
    expected_kinds = {config: ("training_curves", "confusion_matrix", "gradcam_examples")
                      for config in EXPERIMENTS.values()}
    expected_kinds.update({config: ("training_curves",) for config in TUNING_RUNS})
    rows, problems = [], []
    for config, kinds in expected_kinds.items():
        for kind in kinds:
            path = directory / f"{output_stem(config)}_{kind}.png"
            expected = run_title(config)
            if not path.exists():
                state, found = "missing", ""
            else:
                found = read_title(path) or ""
                # The run's own title must appear in the figure's title; the kind may follow it.
                state = "ok" if expected in found else "mislabelled"
            row = {"figure": path.name, "state": state, "expected": expected, "found": found}
            rows.append(row)
            if state != "ok":
                problems.append(row)
    return rows, problems


def synthetic_rows():
    """Build 27 believable metrics dicts, so the report figures can be checked before any run."""
    rng = np.random.default_rng(7)
    rows = []
    for run_id, config in EXPERIMENTS.items():
        # More data helps; strong augmentation helps a little at 20% and hurts slightly at 100%.
        size = int(LOG_LABELS[config.split].rstrip("%"))
        base = 0.93 + 0.04 * np.log10(size / 20 + 1) + rng.normal(0, 0.004)
        regime_shift = {"none": 0.0, "mild": 0.004, "strong": 0.012 - 0.00012 * size}
        value = min(0.999, base + regime_shift[config.augmentation])
        stability_out = 0.72 + 0.05 * (config.augmentation != "none") + rng.normal(0, 0.01)
        rows.append({
            "experiment_id": run_id, "model": LOG_LABELS[config.model],
            "data_size": LOG_LABELS[config.split], "augmentation": LOG_LABELS[config.augmentation],
            "roc_auc": value, "roc_auc_ci_low": value - 0.012, "roc_auc_ci_high": min(1.0, value + 0.012),
            "f1": value - 0.03, "f1_ci_low": value - 0.045, "f1_ci_high": value - 0.015,
            "gradcam_stability_out_of_family": stability_out,
            "gradcam_stability_in_family": stability_out + 0.06 * (config.augmentation == "strong"),
        })
    return rows


def main():
    """Self-check: draw every figure from synthetic data, then audit the titles."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit", action="store_true", help="check the figures under the output root")
    parser.add_argument("--output-root", type=Path, default=None, help="where results/figures lives")
    args = parser.parse_args()

    if args.audit:
        rows, problems = audit(args.output_root)
        for row in problems[:10]:
            print(f"  {row['figure']:34s} {row['state']}")
        print(f"{len(rows) - len(problems)} of {len(rows)} figures present and correctly titled")
        return

    rows = synthetic_rows()
    drawn = report_figures(rows, args.output_root)
    print(f"  report figures: {[path.name for path in drawn]}")
    directory = figures_directory(args.output_root)
    config = EXPERIMENTS["E01"]
    counts = {"tn": 180, "fp": 54, "fn": 12, "tp": 378}
    confusion_figure(counts, f"{run_title(config)} - test confusion matrix",
                     directory / "E01_confusion_matrix.png")
    history = [{"epoch": epoch, "train_loss": 0.6 / epoch, "val_loss": 0.4 + 0.05 * epoch,
                "roc_auc": 0.95 + 0.004 * min(epoch, 4), "balanced_accuracy": 0.80 + 0.01 * epoch}
               for epoch in range(1, 9)]
    curves_figure(history, f"{run_title(config)} - training curves",
                  directory / "E01_training_curves.png")
    written = read_title(directory / "E01_confusion_matrix.png")
    print(f"  title stored in the PNG: {written!r}")
    print(f"  audit finds it correctly labelled: {run_title(config) in (written or '')}")
    rows, problems = audit(args.output_root)
    # Only the two figures drawn above exist, so everything else is expected to be missing -
    # which is also a check that the audit counts all 27 experiments and 9 tuning runs.
    print(f"  audit: {len(rows) - len(problems)} of {len(rows)} present "
          f"(expected 2 of {3 * 27 + 9})")


if __name__ == "__main__":
    main()
