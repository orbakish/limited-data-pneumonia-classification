"""Where the chest X-ray image files live, and how their filenames encode patient identity.

The one module that knows the dataset's folder layout and naming conventions. The survey
script, the split builder and the Dataset all import from here, so they cannot drift into
disagreeing about what counts as a patient - a disagreement that would leak patients across
the train/validation boundary and silently inflate every validation score.

Not a script; import it:

    from src.paths import data_root, group_key, list_images, patient_key
"""

import os
import re
from pathlib import Path

# Filename patterns. PNEUMONIA: person<N>_<bacteria|virus>_<M>[_<K>]. NORMAL: [NORMAL2-]IM-<A>-<B>[-<C>].
# (?:...) groups without capturing (group numbers stay the same); the ? after it makes it optional.
PNEUMONIA_NAME = re.compile(r"person(\d+)_(bacteria|virus)_(\d+)(?:_\d+)?", re.IGNORECASE)
NORMAL_NAME = re.compile(r"(NORMAL2-)?IM-(\d+)-(\d+)(?:-\d+)?", re.IGNORECASE)

# The dataset's folder layout is data_root()/<split>/<class>/*.jpeg
SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}

# Path(__file__) is this file; .resolve() makes it absolute, and .parents[1] is the repo root
# (.parents[0] would be src/). Deriving the default this way means the code finds the data no
# matter which directory python was started from.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "chest_xray"

# Name of the environment variable that overrides the default. Colab unpacks the dataset to
# /content, which is outside the repo, so the location has to be injectable rather than fixed.
DATA_ROOT_ENV = "PNEUMONIA_DATA_ROOT"


def data_root():
    """Return the directory holding the split folders, as a Path.

    Uses $PNEUMONIA_DATA_ROOT when it is set, otherwise the repo's own data/raw/chest_xray.
    """
    # os.environ.get returns None for an unset variable instead of raising, which is what
    # makes the one-line fallback below safe.
    override = os.environ.get(DATA_ROOT_ENV)
    return Path(override) if override else DEFAULT_DATA_ROOT


def list_images(folder):
    """Return the image files directly inside 'folder', sorted by name.

    Skips hidden files, because macOS zip archives add '._<name>.jpeg' metadata files that
    match the image extension but are not images.
    """
    # iterdir() yields every entry in the folder, one level deep only (no recursion).
    # sorted() because iterdir() order depends on the filesystem, and a fixed order is what
    # lets a seeded shuffle produce the same subsets on every machine.
    return sorted(
        path
        for path in folder.iterdir()
        # .suffix is the extension including the dot, e.g ".jpeg"; .lower() also accepts ".JPEG".
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")
    )


def collect_images(root):
    """Return a nested dict {split: {class: [image paths]}} covering every split and class."""
    # The / operator on a path joins path segments, the same way on Windows and Linux.
    return {
        split: {cls: list_images(root / split / cls) for cls in CLASSES}
        for split in SPLITS
    }


def patient_key(cls, path):
    """Return a candidate patient ID parsed from the filename, or None if no pattern matches.

    PNEUMONIA: bacterial and viral cases are numbered in separate sequences, so the key is the
    number plus the infection type. NORMAL: the NORMAL2- prefix stays in the key to avoid
    merging two naming schemes that may number independently.
    """
    # .stem is the filename without its extension: "person1_virus_6.jpeg" -> "person1_virus_6".
    # fullmatch() only succeeds if the pattern covers the entire string, not just part of it.
    if cls == "PNEUMONIA":
        match = PNEUMONIA_NAME.fullmatch(path.stem)
        # group(1) is the patient number, group(2) the infection type. Both go in the key, because
        # person53_bacteria and person53_virus are different patients.
        return f"person{match.group(1)}_{match.group(2).lower()}" if match else None
    match = NORMAL_NAME.fullmatch(path.stem)
    # group(1) is None when the optional NORMAL2- prefix is absent; `or ""` turns that into "".
    return f"{match.group(1) or ''}IM-{match.group(2)}" if match else None

def group_key(cls, path):
    """Return the key that decides which images must stay together when splitting, or None.

    Deliberately coarser than patient_key for PNEUMONIA: it groups by person<N> alone, merging
    person53_bacteria and person53_virus into one group. The evidence says those are different
    children, numbered in separate sequences (survey_dataset.py prints the numbers), but
    merging them can never put one child on both sides of the split whichever reading is
    right, while keeping them apart would leak if the evidence were wrong. NORMAL groups are
    the same as patient_key.
    """
    if cls == "PNEUMONIA":
        match = PNEUMONIA_NAME.fullmatch(path.stem)
        # group(1) alone - the number without the infection type.
        return f"person{match.group(1)}" if match else None
    return patient_key(cls, path)
