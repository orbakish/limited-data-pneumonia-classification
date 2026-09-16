"""Builds every split the experiments use and writes them out as explicit file lists.

Owns two decisions that share a seed and the patient keys, which is why they sit in one
module rather than two:

1. Our own validation split. The official val/ folder holds 16 images, far too few for early
   stopping, so train/ and val/ are combined and about 15% is carved back out - stratified by
   class, grouped by patient. The official test/ split is never touched.
2. The nested 20 / 50 / 100% training subsets, taken as prefixes of a single seeded per-class
   shuffle of whatever remains.

The lists are written to data/splits/, one path per line relative to data_root(), and they are
committed. They are the definition of the experiment, and regenerating them must produce a
byte-identical file - that diff is the reproducibility test.

Regenerate them with:

    python -m src.splits
"""

import random
from collections import defaultdict
from pathlib import Path

from src.paths import CLASSES, DATA_ROOT_ENV, collect_images, data_root, group_key

# Fixed once. Every split in the project derives from this one number, so a single seed
# reproduces the entire experimental design.
SEED = 42

# Fraction of the combined train+val pool held back as our own validation set.
VAL_FRACTION = 0.15

# The training sizes the experiment compares, as fractions of the post-split pool. 1.00 is
# what the report calls "100%" - the pool, not the original Kaggle train/ folder.
SUBSET_FRACTIONS = (0.20, 0.50, 1.00)

SPLIT_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"


def group_images(paths, cls):
    """Group one class's image paths by their split group key (see src.paths.group_key).

    Returns {key: [paths]}. A filename matching no pattern gets a group of its own, keyed by
    the filename, so an unrecognised name is never silently merged into another group.
    """
    # defaultdict(list) creates an empty list the first time a key is used, which removes the
    # "if key not in groups" bookkeeping a plain dict would need.
    groups = defaultdict(list)
    for path in paths:
        key = group_key(cls, path)
        # `key or ...` falls back when group_key returned None. Giving the image its own group is
        # the safe assumption: merging two unrecognised names could just as easily put one real
        # patient on both sides of the split.
        groups[key or f"unmatched:{path.name}"].append(path)
    return groups


def carve_validation(groups, val_fraction, seed_text):
    """Split one class's patient groups into (validation paths, training paths).

    Whole groups move together, so no patient lands on both sides. Groups are drawn in a
    seeded random order until validation reaches val_fraction of the class's images, so the
    result is reproducible without being biased by filename order.
    """
    # Sort before shuffling: the shuffle is only reproducible relative to the order it is given.
    keys = sorted(groups)
    random.Random(seed_text).shuffle(keys)

    target = round(sum(len(groups[key]) for key in keys) * val_fraction)

    validation, training = [], []
    for key in keys:
        # Whole group at a time, so the overshoot past the target is at most one patient's
        # worth of images - the price of never splitting a patient.
        if len(validation) < target:
            validation.extend(groups[key])
        else:
            training.extend(groups[key])
    return sorted(validation), sorted(training)


def nested_subsets(pool, fractions, seed_text):
    """Return {fraction: paths}, each a prefix of one seeded shuffle of 'pool'.

    Prefixes of a single ordering is what makes the subsets nested by construction: the 20%
    list is literally the first part of the 50% list, not a separately drawn sample.
    """
    shuffled = sorted(pool)
    random.Random(seed_text).shuffle(shuffled)
    return {fraction: shuffled[:round(len(shuffled) * fraction)] for fraction in fractions}


def write_split(name, paths, root):
    """Write one split's file list to data/splits/<name>.txt, one path per line.

    Paths are stored relative to 'root' with forward slashes, so the committed file is
    identical on Windows and Linux and stays valid on Colab where the dataset lives elsewhere.
    """
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    # .relative_to(root) strips the machine-specific prefix; .as_posix() forces forward
    # slashes so Windows does not write backslashes into a file Colab has to read.
    lines = [path.relative_to(root).as_posix() for path in paths]
    # newline="\n" matters: without it, write_text translates \n to the platform line ending,
    # and the "regenerating gives a byte-identical file" test would fail across machines.
    (SPLIT_DIR / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def load_split(name, root=None):
    """Read one split's file list and return it as a list of absolute Paths.

    name is a stem such as "val" or "train_20"; root defaults to data_root(). Raises
    FileNotFoundError if any listed image is missing under root: the lists name specific
    files, so a dataset that changed after they were generated must fail here rather than
    train silently on fewer images than the list claims.
    """
    root = root or data_root()
    text = (SPLIT_DIR / f"{name}.txt").read_text(encoding="utf-8")
    # splitlines() copes with both \n and \r\n; the `if line` drops the empty string that the
    # trailing newline leaves behind.
    paths = [root / line for line in text.splitlines() if line]

    # One filesystem check per image - a few thousand take well under a second on a local disk.
    missing = [path for path in paths if not path.is_file()]
    # raise rather than assert: `python -O` strips assert statements, and this check must
    # never be optimised away.
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(paths)} images in split '{name}' are missing under {root}, "
            f"e.g. {missing[0]}. First check that this is the dataset root (set {DATA_ROOT_ENV} "
            f"if not). Regenerate the splits only if the dataset itself changed - that rewrites "
            f"the committed definition of the experiment: python -m src.splits"
        )

    return paths


def check_group_disjoint(train_paths, val_paths):
    """Raise AssertionError if any split group has images in both the training and validation lists.

    The entire point of grouped splitting. A patient on both sides lets the model key on body
    habitus or positioning rather than pathology, and the validation score stops meaning
    anything. Checked with group_key, the same key the split was built with.
    """
    def keys(paths):
        # The class is the name of the folder the image sits in, since the layout is
        # <split>/<class>/<file>: .parent is the containing folder, .name its last component.
        return {group_key(path.parent.name, path) or path.name for path in paths}

    overlap = keys(train_paths) & keys(val_paths)
    assert not overlap, f"{len(overlap)} groups in both train and validation, e.g. {sorted(overlap)[:5]}"


def check_nesting(subset_paths):
    """Raise AssertionError unless each subset is fully contained in the next larger one.

    subset_paths maps each fraction to its list of paths. Nesting is what makes a difference
    between data sizes attributable to *adding* data rather than to drawing a different sample.
    """
    fractions = sorted(subset_paths)
    # zip(seq, seq[1:]) pairs each item with the one after it: (0.2, 0.5), then (0.5, 1.0).
    for smaller, larger in zip(fractions, fractions[1:]):
        missing = set(subset_paths[smaller]) - set(subset_paths[larger])
        assert not missing, f"{len(missing)} images in the {smaller:.0%} subset are absent from {larger:.0%}"


def build_splits(seed=SEED, val_fraction=VAL_FRACTION, root=None):
    """Generate every split and write the file lists to data/splits/.

    Returns {split name: paths} for everything it wrote. Raises AssertionError if a patient
    spans train and validation, or if the subsets are not nested.
    """
    root = root or data_root()
    images = collect_images(root)

    validation = []
    subsets = {fraction: [] for fraction in SUBSET_FRACTIONS}
    for cls in CLASSES:
        # Official train and val are pooled first: the official val split is 8 images per
        # class, far too few to select a model on. Sorted so the pool does not depend on which
        # order the two folders happened to be concatenated in.
        pool = sorted(images["train"][cls] + images["val"][cls])

        # Done per class, so the NORMAL:PNEUMONIA ratio is preserved on both sides.
        cls_val, cls_train = carve_validation(group_images(pool, cls), val_fraction, f"{seed}:val:{cls}")
        validation.extend(cls_val)

        for fraction, paths in nested_subsets(cls_train, SUBSET_FRACTIONS, f"{seed}:subset:{cls}").items():
            subsets[fraction].extend(paths)

    check_group_disjoint(subsets[1.00], validation)
    check_nesting(subsets)

    written = {"val": sorted(validation), "test": sorted(images["test"]["NORMAL"] + images["test"]["PNEUMONIA"])}
    for fraction, paths in subsets.items():
        # int(fraction * 100) turns 0.2 into the name train_20.
        written[f"train_{int(fraction * 100)}"] = sorted(paths)

    for name, paths in written.items():
        write_split(name, paths, root)
    return written


def main():
    written = build_splits()
    print(f"--- splits written to {SPLIT_DIR} ---")
    for name, paths in written.items():
        normal = sum(1 for path in paths if path.parent.name == "NORMAL")
        pneumonia = len(paths) - normal
        ratio = f"{pneumonia / normal:.2f}" if normal else "n/a"
        print(f" {name:10s} {len(paths):5d}  NORMAL {normal:5d}  PNEUMONIA {pneumonia:5d}  ratio {ratio}")


if __name__ == "__main__":
    main()
