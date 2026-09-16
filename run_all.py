"""Run every experiment, E01 through E27, or the nine tuning runs.

A thin orchestrator: it does not know how to train anything, it just calls the same code path
run_experiment.py uses, once per config. Configs whose results already exist are skipped, so a
dropped Colab session costs one experiment rather than the whole sweep.

    python run_all.py
    python run_all.py --tuning          # T01-T09 instead of the sweep
    python run_all.py --force           # re-run configs that already have results
    python run_all.py --workers 2       # DataLoader workers, worth 2 on Colab

Finished runs are recognised by their outputs: an experiment by its metrics file, a tuning run
by history.json saying the run finished. Both live under $PNEUMONIA_OUTPUT_ROOT.
"""

import argparse
import json
from pathlib import Path

from run_experiment import run
from src.config import EXPERIMENTS, TUNING_RUNS, get_config, output_stem
from src.evaluate import results_directory
from src.paths import OUTPUT_ROOT_ENV
from src.splits import SEED
from src.train import run_directory


def parse_args():
    """Parse the command-line arguments and return them as a Namespace."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # action="store_true" makes --force a flag: present means True, absent means False, and it
    # takes no value of its own.
    parser.add_argument("--force", action="store_true", help="re-run configs whose results already exist")
    parser.add_argument("--tuning", action="store_true", help="run T01-T09 instead of E01-E27")
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default: {SEED})")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (default: 0)")
    parser.add_argument("--resume-every", type=int, default=1,
                        help="write the resume checkpoint every N epochs (default: 1)")
    parser.add_argument("--output-root", type=Path, default=None, help=f"default: ${OUTPUT_ROOT_ENV}, else the repo")
    return parser.parse_args()


def already_done(run_id, seed=SEED, root=None):
    """Return True if this run's outputs already exist.

    An experiment is done when its metrics file exists; a tuning run when its history says the
    run finished, because a tuning run writes no file under results/metrics/. The name carries
    the seed when it is not the default, so a second-seed sweep is not skipped as done.
    """
    config = get_config(run_id, seed)
    if run_id.startswith("E"):
        return (results_directory(root) / f"{output_stem(config)}_metrics.json").exists()
    history = run_directory(config, root) / "history.json"
    return history.exists() and json.loads(history.read_text(encoding="utf-8"))["finished"]


def main():
    """Run every config in order, skipping the ones that already have results."""
    args = parse_args()
    run_ids = [config.run_id for config in TUNING_RUNS] if args.tuning else list(EXPERIMENTS)
    for run_id in run_ids:
        if not args.force and already_done(run_id, args.seed, args.output_root):
            print(f"{run_id}: already done, skipping")
            continue
        print(f"{run_id}: starting")
        run(run_id, args.seed, args.workers, args.output_root, resume_every=args.resume_every)


if __name__ == "__main__":
    main()
