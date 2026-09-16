"""Architecture-agnostic training loop for one run, resumable after an interruption.

train(config) trains the run's model on its training split, scores the validation set after
every epoch, keeps the checkpoint with the best validation ROC-AUC, and stops after
RECIPE.patience epochs without improvement or at RECIPE.max_epochs. It never checks which
architecture it was given - everything model-specific goes through src/models.py.

Each run writes to <output root>/checkpoints/<run_id>_seed<seed>/:

* best.pt - the best epoch's weights with the config, recipe, metrics and library versions.
  This is what evaluation and Grad-CAM load.
* last.pt - weights and optimizer state after the latest epoch, so an interrupted run continues
  where it stopped. Deleted once the run finishes.
* history.json - one row of numbers per epoch, and whether the run has finished.
* train.log - every progress message printed during training, appended across resumes.

The output root is $PNEUMONIA_OUTPUT_ROOT (the Colab notebook points it at Drive), otherwise the
repo itself. Resuming gives the same result as never stopping, because each epoch's randomness
- shuffle order, augmentation, dropout - is seeded from (seed, epoch) rather than carried over
from the epoch before.

    python -m src.train --id T01
    python -m src.train --id T01 --max-epochs 2 --output-root C:/temp/smoke   # smoke test
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn

from src.augment import build_transform
from src.config import RECIPE, get_config
from src.data import build_dataloader
from src.evaluate import predict, scores
from src.models import build_model, set_fine_tuning, split_parameters, train_mode

OUTPUT_ROOT_ENV = "PNEUMONIA_OUTPUT_ROOT"
REPO_ROOT = Path(__file__).resolve().parents[1]


def output_root():
    """Return the folder runs are written under: $PNEUMONIA_OUTPUT_ROOT if set, else the repo."""
    override = os.environ.get(OUTPUT_ROOT_ENV)
    return Path(override) if override else REPO_ROOT


def run_directory(config, root=None):
    """Return the folder holding one run's checkpoints, history and log."""
    return (root or output_root()) / "checkpoints" / f"{config.run_id}_seed{config.seed}"


def get_device():
    """Return the torch device to train on, plus a readable name for it."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    return torch.device("cpu"), "CPU"


def seed_everything(seed):
    """Seed Python's, NumPy's and torch's random generators; torch's seed covers every GPU too."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epoch_seed(seed, epoch):
    """Return the seed for one epoch - different for every (seed, epoch) pair below 1000 epochs."""
    return seed * 1000 + epoch


def class_weights(labels):
    """Return inverse-frequency weights N / (2 * n_class) as a tensor [NORMAL, PNEUMONIA].

    A class with half the images gets twice the weight, so both classes contribute equally to
    the loss in total.
    """
    # bincount counts how often each integer occurs: index 0 holds the NORMAL count, 1 PNEUMONIA.
    counts = torch.bincount(torch.tensor(labels), minlength=2)
    return len(labels) / (2 * counts.float())


def save_atomically(payload, path):
    """Save with torch.save to a temporary file, then rename it over path.

    A disconnect in the middle of torch.save would leave a truncated file; a rename either
    happens completely or not at all, so path always holds the last complete save.
    """
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    # Path.replace renames, overwriting the target if it exists.
    temporary.replace(path)


def write_history(path, history, finished):
    """Write the per-epoch rows and the finished flag to history.json, atomically."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps({"finished": finished, "epochs": history}, indent=2), encoding="utf-8")
    temporary.replace(path)


def open_logger(run_dir):
    """Return a logger that prints each message and also appends it to run_dir/train.log."""
    logger = logging.getLogger(f"train.{run_dir.name}")
    logger.setLevel(logging.INFO)
    # Without this, messages would also travel up to Python's root logger and could print twice.
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # A handler decides where messages go: one to the terminal, one to the file (mode "a" appends).
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log", mode="a", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger):
    """Detach and close the logger's handlers, releasing train.log."""
    # Iterate over a copy, because removeHandler changes the list being looped over.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def train_one_epoch(model, config, loader, criterion, optimizer, device):
    """Run one pass over the training data, updating the weights, and return the mean batch loss."""
    train_mode(model, config.model, config.fine_tuning)
    total = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        # Gradients add up across backward() calls, so they are cleared before each batch.
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        # .item() copies a one-value tensor out as a plain Python number.
        total += loss.item()
    return total / len(loader)


def log_header(logger, config, recipe, device_name, model, train_labels, val_size, weights):
    """Log what this run is: its config, recipe, device, data and trainable parameters."""
    normal, pneumonia = train_labels.count(0), train_labels.count(1)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"run {config.run_id}: {config.model}, {config.split}, augmentation {config.augmentation}, "
                f"{config.fine_tuning} fine-tuning, weighted loss {config.weighted_loss}, seed {config.seed}")
    logger.info(f"recipe: {asdict(recipe)}")
    logger.info(f"device {device_name}, torch {torch.__version__}, torchvision {torchvision.__version__}, "
                f"cudnn deterministic {torch.backends.cudnn.deterministic}")
    weight_text = f"class weights NORMAL {weights[0]:.2f} PNEUMONIA {weights[1]:.2f}" if weights is not None else "unweighted loss"
    logger.info(f"data: {config.split} {len(train_labels)} images (NORMAL {normal}, PNEUMONIA {pneumonia}), "
                f"val {val_size} images, {weight_text}")
    logger.info(f"parameters: {trainable:,} trainable of {total:,}")


def fit(config, recipe, run_dir, logger, num_workers):
    """Train one run inside run_dir, resuming from last.pt if present. Returns the history rows."""
    history_path, last_path, best_path = (run_dir / name for name in ("history.json", "last.pt", "best.pt"))
    device, device_name = get_device()
    # Deterministic cuDNN convolutions, and no benchmarking (which picks algorithms by timing
    # them, so the choice can differ between runs). AdaptiveAvgPool2d's backward pass on the
    # GPU stays non-deterministic regardless, so reruns are close rather than bit-identical.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Seeded before the model is built, so the new head's random initial weights are reproducible.
    seed_everything(config.seed)
    model = build_model(config.model).to(device)
    set_fine_tuning(model, config.model, config.fine_tuning)
    head, backbone = split_parameters(model, config.model)
    # Two parameter groups, each with its own learning rate; weight decay applies to both.
    optimizer = torch.optim.AdamW(
        [{"params": head, "lr": recipe.head_lr}, {"params": backbone, "lr": recipe.backbone_lr}],
        weight_decay=recipe.weight_decay,
    )

    train_loader = build_dataloader(config.split, recipe.batch_size, build_transform(config.augmentation, train=True),
                                    shuffle=True, seed=config.seed, num_workers=num_workers)
    val_loader = build_dataloader("val", recipe.batch_size, build_transform(config.augmentation, train=False),
                                  num_workers=num_workers)
    train_labels = train_loader.dataset.labels
    weights = class_weights(train_labels) if config.weighted_loss else None
    # The validation loss uses this same criterion, so it is weighted too when the run is.
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    log_header(logger, config, recipe, device_name, model, train_labels, len(val_loader.dataset), weights)

    state = {"epoch": 0, "best_epoch": 0, "best_roc_auc": float("-inf"), "history": []}
    if last_path.exists():
        # map_location loads tensors straight onto this machine's device, whatever device saved them.
        checkpoint = torch.load(last_path, map_location=device)
        if checkpoint["config"] != asdict(config) or checkpoint["recipe"] != asdict(recipe):
            raise RuntimeError(f"{last_path} was written with a different config or recipe - delete it to start over")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        state = checkpoint["state"]
        logger.info(f"resuming after epoch {state['epoch']} "
                    f"(best val ROC-AUC {state['best_roc_auc']:.4f} at epoch {state['best_epoch']})")
    else:
        logger.info("starting fresh")
    stopped_by = "epoch cap"
    while state["epoch"] < recipe.max_epochs:
        if state["epoch"] - state["best_epoch"] >= recipe.patience:
            stopped_by = "patience"
            break
        epoch = state["epoch"] + 1
        started = time.perf_counter()
        # Reseed per epoch: the global generators drive augmentation and dropout, and the loader's
        # own generator drives the shuffle order and worker seeds.
        seed_everything(epoch_seed(config.seed, epoch))
        train_loader.generator.manual_seed(epoch_seed(config.seed, epoch))

        train_loss = train_one_epoch(model, config, train_loader, criterion, optimizer, device)
        logits, labels = predict(model, val_loader, device)
        with torch.no_grad():
            val_loss = criterion(logits.to(device), labels.to(device)).item()
        metrics = scores(logits, labels)

        improved = metrics["roc_auc"] > state["best_roc_auc"]
        if improved:
            state["best_roc_auc"], state["best_epoch"] = metrics["roc_auc"], epoch
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics,
               "seconds": round(time.perf_counter() - started, 1)}
        state["history"].append(row)
        state["epoch"] = epoch

        if improved:
            save_atomically({"model": model.state_dict(), "config": asdict(config), "recipe": asdict(recipe),
                             "epoch": epoch, "metrics": row,
                             # str() because torch.__version__ is a special subclass of str that
                             # torch.load refuses to unpickle by default.
                             "versions": {"torch": str(torch.__version__), "torchvision": torchvision.__version__}},
                            best_path)
        save_atomically({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state,
                         "config": asdict(config), "recipe": asdict(recipe)}, last_path)
        write_history(history_path, state["history"], finished=False)

        since_best = "* best" if improved else f"{epoch - state['best_epoch']}/{recipe.patience} without improvement"
        logger.info(f"epoch {epoch:2d}/{recipe.max_epochs}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
                    f"val ROC-AUC {metrics['roc_auc']:.4f}  bal acc {metrics['balanced_accuracy']:.4f}  "
                    f"{since_best}  ({row['seconds']:.0f}s)")

    write_history(history_path, state["history"], finished=True)
    last_path.unlink()
    best = state["history"][state["best_epoch"] - 1]
    minutes = sum(row["seconds"] for row in state["history"]) / 60
    logger.info(f"finished: stopped by {stopped_by} after epoch {state['epoch']}; best epoch {best['epoch']} "
                f"val ROC-AUC {best['roc_auc']:.4f} bal acc {best['balanced_accuracy']:.4f}; "
                f"{minutes:.1f} min of training")
    return state["history"]


def train(config, recipe=RECIPE, root=None, num_workers=0):
    """Train one run end to end, or finish an interrupted one, and return its run folder.

    A run whose history.json already says finished is skipped. num_workers > 0 decodes images in
    background processes, which speeds up epochs on Colab.
    """
    run_dir = run_directory(config, root)
    history_path = run_dir / "history.json"
    if history_path.exists() and json.loads(history_path.read_text(encoding="utf-8"))["finished"]:
        print(f"{config.run_id}: already finished - {run_dir}")
        return run_dir
    # parents=True creates missing parent folders; exist_ok=True accepts a folder left by an earlier attempt.
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = open_logger(run_dir)
    # try/finally releases train.log even if training raises, e.g. on a keyboard interrupt.
    try:
        fit(config, recipe, run_dir, logger, num_workers)
    finally:
        close_logger(logger)
    return run_dir


def parse_args():
    """Parse the command-line arguments and return them as a Namespace."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True, help="run ID, E01-E27 or T01-T09")
    parser.add_argument("--seed", type=int, default=None, help="override the run's seed")
    parser.add_argument("--max-epochs", type=int, default=None, help="override RECIPE.max_epochs, for smoke tests")
    parser.add_argument("--output-root", type=Path, default=None, help=f"default: ${OUTPUT_ROOT_ENV}, else the repo")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (default: 0)")
    return parser.parse_args()


def main():
    """Train the run named on the command line."""
    args = parse_args()
    recipe = RECIPE if args.max_epochs is None else replace(RECIPE, max_epochs=args.max_epochs)
    train(get_config(args.id, seed=args.seed), recipe, args.output_root, args.workers)


if __name__ == "__main__":
    main()
