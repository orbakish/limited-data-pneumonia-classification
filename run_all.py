"""Run every experiment, E01 through E27.

A thin orchestrator: it does not know how to train anything, it just calls the same code path
run_experiment.py uses, once per config. Configs whose metrics file already exists are skipped,
so a dropped Colab session costs one experiment rather than the whole sweep.

    python run_all.py
    python run_all.py --force     # re-run configs that already have results
"""

import argparse

from run_experiment import run
from src.splits import SEED

# 3 models x 3 data sizes x 3 augmentation regimes, numbered as in experiments/experiment_log.md.
# f"E{n:02d}" zero-pads to two digits, so 1 becomes "E01" rather than "E1".
EXPERIMENT_IDS = tuple(f"E{number:02d}" for number in range(1, 28))

def parse_args():
    """Parse the command-line arguments and return them as a Namespace."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # action="store_true" makes --force a flag: present means True, absent means False, and it
    # takes no value of its own.
    parser.add_argument("--force", action="store_true", help="re-run configs whose results already exist")
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default: {SEED})")
    return parser.parse_args()

def already_done(experiment_id):
    """Return True if this experiment's metrics file already exists."""
    raise NotImplementedError("needs the results path from src/config.py - workplan section 3")

def main():
    args = parse_args()
    for experiment_id in EXPERIMENT_IDS:
        if not args.force and already_done(experiment_id):
            print(f"{experiment_id}: already done, skipping")
            continue
        print(f"{experiment_id}: starting")
        run(experiment_id, args.seed)

if __name__ == "__main__":
    main()
