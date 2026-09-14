"""Run a single experiment, identified by its E01-E27 ID.

The only place an experiment is launched from. run_all.py loops the 27 configs calling this
same code path, so anything that runs in the sweep can also be run, debugged or re-run on its
own - which matters because Colab sessions drop mid-sweep.

    python run_experiment.py --id E01
    python run_experiment.py --id E01 --seed 7

Writes metrics to results/metrics/E##_metrics.json and figures to results/figures/, per the
schema in results/README.md.
"""

import argparse

from src.splits import SEED

def parse_args():
    """Parse the command-line arguments and return them as a Namespace."""
    # RawDescriptionHelpFormatter stops argparse from re-wrapping the docstring above into one
    # paragraph when --help prints it.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--id", required=True, help="experiment ID, E01 through E27")
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed (default: {SEED})")
    return parser.parse_args()

def run(experiment_id, seed):
    """Train, evaluate, explain and score one config end to end.

    Returns the metrics dict that was written to results/metrics/.
    """
    raise NotImplementedError("needs src/config.py, src/train.py, src/evaluate.py - workplan sections 3-7")

def main():
    args = parse_args()
    run(args.id, args.seed)

if __name__ == "__main__":
    main()

