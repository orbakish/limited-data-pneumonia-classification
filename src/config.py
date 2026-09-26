"""Run configurations: the 27 experiments E01-E27, the sixteen tuning runs T01-T16, and the recipe.

A run is fully described by a RunConfig - which model, which training subset, which
augmentation regime, how much of the network trains, whether the loss is class-weighted, and
the seed - plus the RECIPE that is identical for every run (NOTES.md #18). Both are saved with
every result, so each number can be traced back to exactly what produced it.

The E-IDs follow experiments/experiment_log.md: model first, then data size, then augmentation,
so E01 is MobileNetV3-Small / 20% / none and E27 is EfficientNet-B0 / 100% / strong. The T-IDs
follow the tuning-phase protocol in NOTES.md, validation only, on train_20: T01-T09 without
augmentation, T10-T15 a spot-check with strong augmentation, and T16 a BatchNorm diagnostic.

Import it:

    from src.config import RECIPE, get_config
    config = get_config("E07")            # or "T04"
    config = get_config("E07", seed=7)    # same run, another seed

or run it to print every run and check the E-IDs against experiments/experiment_log.md:

    python -m src.config
"""

import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from src.augment import REGIMES
from src.models import ARCHITECTURES, FINE_TUNING
from src.splits import SEED, SUBSET_FRACTIONS

EXPERIMENT_LOG = Path(__file__).resolve().parents[1] / "experiments" / "experiment_log.md"

# The order of these three tuples defines the E-IDs, so it mirrors experiment_log.md rather
# than relying on the order of ARCHITECTURES or REGIMES in other modules.
MODELS = ("mobilenet_v3_small", "resnet18", "efficientnet_b0")
TRAIN_SPLITS = tuple(f"train_{int(fraction * 100)}" for fraction in SUBSET_FRACTIONS)
AUGMENTATIONS = ("none", "mild", "strong")

# How experiment_log.md writes each value, for the self-check.
LOG_LABELS = {
    "mobilenet_v3_small": "MobileNetV3-Small", "resnet18": "ResNet18",
    "efficientnet_b0": "EfficientNet-B0",
    "train_20": "20%", "train_50": "50%", "train_100": "100%",
    "none": "None", "mild": "Mild", "strong": "Strong",
}

# Decided by the tuning phase (NOTES.md #6, #17, 2026-09-26): partial fine-tuning, because it was
# never more than 0.002 behind full on any architecture, with or without augmentation, and one
# recipe applies to all three. Its clear lead on MobileNetV3-Small came from BatchNorm running
# statistics that lag under full fine-tuning at torchvision's default momentum of 0.01, not from
# worse weights (T16, NOTES.md #26); partial avoids that lag with the standard settings. The
# class-weighted loss stays, because leaving it out made no detectable difference.
DEFAULT_FINE_TUNING = "partial"
DEFAULT_WEIGHTED_LOSS = True


# frozen=True makes assigning to a field after creation raise an error, so a config cannot be
# changed part-way through a run by accident.
@dataclass(frozen=True)
class Recipe:
    """Training settings shared by every run (NOTES.md #18): AdamW with a constant learning rate."""

    head_lr: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 30
    # Early stopping: stop after this many epochs without a better selection metric.
    patience: int = 5
    # Which validation metric picks the saved checkpoint and drives early stopping. Balanced
    # accuracy rather than ROC-AUC, decided 2026-09-16 from T01: ROC-AUC saturated (epochs 7-12
    # all inside its own confidence interval) while balanced accuracy still climbed 0.92 -> 0.95,
    # so selecting on ROC-AUC kept a checkpoint 3 points worse on every threshold metric.
    selection_metric: str = "balanced_accuracy"


RECIPE = Recipe()


@dataclass(frozen=True)
class RunConfig:
    """Everything that distinguishes one run from another."""

    run_id: str
    model: str
    split: str
    augmentation: str
    fine_tuning: str = DEFAULT_FINE_TUNING
    weighted_loss: bool = DEFAULT_WEIGHTED_LOSS
    seed: int = SEED
    # None keeps torchvision's own BatchNorm momentum. Set only by the T16 diagnostic (NOTES.md #26).
    bn_momentum: float | None = None

    # __post_init__ runs straight after the generated __init__ has filled in the fields.
    def __post_init__(self):
        """Raise ValueError for an unknown model, split, regime or mode, or an out-of-range momentum."""
        allowed = {
            "model": ARCHITECTURES, "split": TRAIN_SPLITS,
            "augmentation": REGIMES, "fine_tuning": FINE_TUNING,
        }
        for field, options in allowed.items():
            value = getattr(self, field)
            if value not in options:
                raise ValueError(f"{self.run_id}: unknown {field} {value!r}, expected one of {list(options)}")
        # None is the default and means "leave torchvision's value", so only a set value is checked.
        if self.bn_momentum is not None and not 0 < self.bn_momentum <= 1:
            raise ValueError(f"{self.run_id}: bn_momentum must be in (0, 1], got {self.bn_momentum}")


def build_experiments():
    """Return the 27 experiments as a dict from E-ID to RunConfig, in experiment_log.md order."""
    experiments = {}
    number = 1
    for model in MODELS:
        for split in TRAIN_SPLITS:
            for augmentation in AUGMENTATIONS:
                # :02d pads the number to two digits, so 1 becomes "01".
                run_id = f"E{number:02d}"
                experiments[run_id] = RunConfig(run_id, model, split, augmentation)
                number += 1
    return experiments


# Written out rather than generated: each block changes exactly one thing against T01-T03, and
# the table should read like the protocol in NOTES.md.
TUNING_RUNS = (
    RunConfig("T01", "mobilenet_v3_small", "train_20", "none", "full", True),
    RunConfig("T02", "resnet18", "train_20", "none", "full", True),
    RunConfig("T03", "efficientnet_b0", "train_20", "none", "full", True),
    RunConfig("T04", "mobilenet_v3_small", "train_20", "none", "partial", True),
    RunConfig("T05", "resnet18", "train_20", "none", "partial", True),
    RunConfig("T06", "efficientnet_b0", "train_20", "none", "partial", True),
    RunConfig("T07", "mobilenet_v3_small", "train_20", "none", "full", False),
    RunConfig("T08", "resnet18", "train_20", "none", "full", False),
    RunConfig("T09", "efficientnet_b0", "train_20", "none", "full", False),
    # Spot-check of the depth decision under strong augmentation (NOTES.md #6): full, then partial.
    RunConfig("T10", "mobilenet_v3_small", "train_20", "strong", "full", True),
    RunConfig("T11", "resnet18", "train_20", "strong", "full", True),
    RunConfig("T12", "efficientnet_b0", "train_20", "strong", "full", True),
    RunConfig("T13", "mobilenet_v3_small", "train_20", "strong", "partial", True),
    RunConfig("T14", "resnet18", "train_20", "strong", "partial", True),
    RunConfig("T15", "efficientnet_b0", "train_20", "strong", "partial", True),
    # A diagnostic, not a recipe candidate: T01 with the BatchNorm momentum ResNet18 and
    # EfficientNet-B0 already use (NOTES.md #26). bn_momentum= names the field, since it comes
    # after seed and the rows above leave seed at its default.
    RunConfig("T16", "mobilenet_v3_small", "train_20", "none", "full", True, bn_momentum=0.1),
)

EXPERIMENTS = build_experiments()
# {**a, **b} builds one dict holding the entries of both.
RUNS = {**EXPERIMENTS, **{config.run_id: config for config in TUNING_RUNS}}


def get_config(run_id, seed=None):
    """Return the RunConfig for an E- or T-ID, optionally with a different seed.

    Raises ValueError for an ID that does not exist.
    """
    if run_id not in RUNS:
        raise ValueError(f"unknown run ID {run_id!r}, expected E01-E27 or T01-T16")
    config = RUNS[run_id]
    # A frozen dataclass cannot be edited, so replace() returns a copy with the seed changed.
    return config if seed is None else replace(config, seed=seed)


def output_stem(config):
    """Return the filename stem for this run's metrics and figures.

    Plain "E01" at the default seed, so the names match the ones results/README.md documents;
    "E01_seed7" otherwise, so a second seed cannot overwrite the first run's results - which is
    what question #21 would do if it ever runs the sweep more than once.
    """
    return config.run_id if config.seed == SEED else f"{config.run_id}_seed{config.seed}"


def run_title(config):
    """Return the one-line label every figure and report table uses for a run."""
    return (f"{config.run_id} - {LOG_LABELS[config.model]} - {LOG_LABELS[config.split]} data - "
            f"augmentation {config.augmentation} - seed {config.seed}")


def check_against_log():
    """Return the E-IDs that are missing from experiment_log.md, extra in it, or differ from it."""
    logged = {}
    for line in EXPERIMENT_LOG.read_text(encoding="utf-8").splitlines():
        # A table row looks like "| E01 | MobileNetV3-Small |       20% | None | ...".
        match = re.match(r"\|\s*(E\d\d)\s*\|([^|]+)\|([^|]+)\|([^|]+)\|", line)
        if match:
            # The * in "run_id, *values" collects every remaining item into a list.
            run_id, *values = (part.strip() for part in match.groups())
            logged[run_id] = values

    mismatched = sorted(set(logged) - set(EXPERIMENTS))
    for run_id, config in EXPERIMENTS.items():
        expected = [LOG_LABELS[config.model], LOG_LABELS[config.split], LOG_LABELS[config.augmentation]]
        # .get returns None for an ID missing from the log, which never equals a list.
        if logged.get(run_id) != expected:
            mismatched.append(run_id)
    return mismatched


def main():
    """Print every run, then check the table against experiment_log.md and the ID lookup."""
    for config in RUNS.values():
        # Printed only when set, so the usual rows stay short and T16 stands out.
        momentum = f" bn_momentum={config.bn_momentum}" if config.bn_momentum is not None else ""
        print(f"  {config.run_id}  {config.model:18s} {config.split:9s} {config.augmentation:6s} "
              f"{config.fine_tuning:7s} weighted={config.weighted_loss}{momentum}")
    print(f"recipe: {asdict(RECIPE)}")
    print(f"experiments: {len(EXPERIMENTS)}, tuning runs: {len(TUNING_RUNS)}")
    print(f"E-IDs differing from experiment_log.md: {check_against_log() or 'none'}")
    reseeded = get_config("E07", seed=7)
    same_otherwise = asdict(reseeded) | {"seed": SEED} == asdict(get_config("E07"))
    print(f"seed override: seed {reseeded.seed}, every other field unchanged: {same_otherwise}")


if __name__ == "__main__":
    main()

