"""Turn the two sweeps into the report's evidence: collect per-image scores, then compare runs.

The results files hold each run's metrics but not its per-image scores, and a difference between
two runs needs both runs scored on the same resampled images. So this module works in two stages:

* collect (Colab, GPU): every experiment's best checkpoint scores validation and test again - the
  exact evaluation path - and the result is checked against the stored results file before it is
  kept. Each run's small files are gathered beside the scores under results/runs/<ID>_seed<N>/, so
  the archive in the repo holds everything the comparisons read.
* analyse (locally, CPU, from the archive): the paired comparisons - part 2 of this module.

    python -m src.analysis --collect --seed 42 --workers 2    # Colab; add --id E01 for one run
    python -m src.analysis --check                             # self-check on synthetic data
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torchvision

from src.config import EXPERIMENTS, RECIPE, get_config, output_stem
from src.evaluate import choose_threshold, confusion_counts, results_directory, score_split, scores
from src.paths import code_version
from src.splits import SEED
from src.train import get_device, load_best, run_directory

# Copied unchanged from a run's checkpoint folder. The raw maps are converted instead, because
# .gitignore keeps every *.pt file out of the repo.
RUN_FILES = ("stability_per_image.json", "validation_metrics.json", "history.json")
# The re-scoring gate: equal to float rounding, or equal up to GPU or library drift.
EXACT, APPROX = 1e-12, 1e-6


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


def self_check():
    """Check the re-scoring gate on synthetic logits: identical, drifted, and one changed prediction."""
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


def main():
    """Collect on Colab, or run the self-check."""
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
        parser.print_help()


if __name__ == "__main__":
    main()
