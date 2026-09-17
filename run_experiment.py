"""Run a single experiment, identified by its E01-E27 or T01-T09 ID.

The only place an experiment is launched from. run_all.py loops the 27 configs calling this
same code path, so anything that runs in the sweep can also be run, debugged or re-run on its
own - which matters because Colab sessions drop mid-sweep.

    python run_experiment.py --id E01
    python run_experiment.py --id E01 --seed 7
    python run_experiment.py --id T01 --workers 2      # tuning run: training only

An E-run trains, then evaluates the best checkpoint on the untouched official test split and
writes results/metrics/E##_metrics.json plus its figures. A T-run stops after training: the
tuning phase is decided on validation alone, and the test set stays untouched until the sweep.

Rerunning an interrupted run resumes it; rerunning a finished one is a no-op.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from src.augment import build_transform
from src.config import RECIPE, get_config, run_title
from src.data import build_dataloader
from src.evaluate import predict, write_metrics
from src.figures import confusion_figure, curves_figure, figures_directory
from src.models import build_model
from src.paths import OUTPUT_ROOT_ENV
from src.train import get_device, train


def parse_args():
    """Parse the command-line arguments and return them as a Namespace."""
    # RawDescriptionHelpFormatter stops argparse from re-wrapping the docstring above into one
    # paragraph when --help prints it.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--id", required=True, help="run ID, E01 through E27 or T01 through T09")
    parser.add_argument("--seed", type=int, default=None, help="override the run's seed")
    parser.add_argument("--max-epochs", type=int, default=None, help="override RECIPE.max_epochs, for smoke tests")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (default: 0)")
    parser.add_argument("--output-root", type=Path, default=None, help=f"default: ${OUTPUT_ROOT_ENV}, else the repo")
    return parser.parse_args()


def evaluate_on_test(config, run_dir, root, workers, recipe):
    """Load the run's best checkpoint, score the test split, and write its metrics and figure."""
    device, _ = get_device()
    # map_location puts the tensors on this machine's device, whatever device saved them.
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    # pretrained=False: the checkpoint replaces every weight, so downloading the ImageNet ones
    # first would be wasted work.
    model = build_model(config.model, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])

    # Test uses preprocessing only - train=False - whatever regime the run trained with.
    loader = build_dataloader("test", recipe.batch_size,
                              build_transform(config.augmentation, train=False),
                              num_workers=workers)
    logits, labels = predict(model, loader, device)
    metrics = write_metrics(config, logits, labels, loader.dataset.paths, root)
    confusion_figure(metrics["confusion_matrix"], f"{run_title(config)} - test confusion matrix",
                     figures_directory(root) / f"{config.run_id}_confusion_matrix.png")
    return metrics


def run(experiment_id, seed=None, workers=0, root=None, recipe=RECIPE):
    """Train, evaluate and plot one config end to end.

    Returns the metrics dict written to results/metrics/, or None for a tuning run, which stops
    after training.
    """
    config = get_config(experiment_id, seed)
    run_dir = train(config, recipe, root, workers)

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))["epochs"]
    curves_figure(history, f"{run_title(config)} - training curves",
                  figures_directory(root) / f"{config.run_id}_training_curves.png")

    if config.run_id.startswith("T"):
        best = max(history, key=lambda row: row["roc_auc"])
        print(f"{config.run_id}: tuning run, validation only - best epoch {best['epoch']}, "
              f"val ROC-AUC {best['roc_auc']:.4f}, balanced accuracy {best['balanced_accuracy']:.4f}, "
              f"F1 {best.get('f1', float('nan')):.4f}")
        return None

    metrics = evaluate_on_test(config, run_dir, root, workers, recipe)
    # TODO workplan sections 6-7: the Grad-CAM examples figure, and the two stability fields,
    # which write_metrics currently leaves null.
    print(f"{config.run_id}: test ROC-AUC {metrics['roc_auc']:.4f} "
          f"({metrics['roc_auc_ci_low']:.4f}-{metrics['roc_auc_ci_high']:.4f}), "
          f"F1 {metrics['f1']:.4f}, balanced accuracy {metrics['balanced_accuracy']:.4f}")
    return metrics


def main():
    """Run the experiment named on the command line."""
    args = parse_args()
    recipe = RECIPE if args.max_epochs is None else replace(RECIPE, max_epochs=args.max_epochs)
    run(args.id, args.seed, args.workers, args.output_root, recipe)


if __name__ == "__main__":
    main()
