"""Dataset survey for the pneumonia classification project.

Measures the dataset properties the input pipeline depends on:
    1. image counts per split and class
    2. image sizes and colour modes
    3. filename patterns (for patient-level splitting)
    4. how much border surrounds the images
Run from the repo root:
    python survey_dataset.py    # local copy
    python survey_dataset.py --data-root /content/chest_xray    # colab

Prints a report. Contact sheets are saved under data/processed/survey/, which is gitignored because they contain patient images.
"""

import argparse
import re
from collections import Counter
from pathlib import Path
from statistics import median

from PIL import Image
from tqdm import tqdm

# Filename patterns. PNEUMONIA: person<N>_<bacteria|virus>_<M>[_<K>]. NORMAL: [NORMAL2-]IM-<A>-<B>[-<C>].
# (?:...) groups without capturing (group numbers stay the same); the ? after it makes it optional.
PNEUMONIA_NAME = re.compile(r"person(\d+)_(bacteria|virus)_(\d+)(?:_\d+)?", re.IGNORECASE)
NORMAL_NAME = re.compile(r"(NORMAL2-)?IM-(\d+)-(\d+)(?:-\d+)?", re.IGNORECASE)
SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}

def list_images(folder):
    """Return the image files directly inside 'folder', sorted by name.
    
    Skips hidden files, because macOS zip archives add '._<name>.jpeg' metadata files that match the image extension but are not images.
    """
    # iterdir() yields every entry in the folder, one level deep only (no recursion).
    # sorted() because iterdir() order depends on the filesystem; and a fixed order is what lets a 
    # seeded shuffle produce the same subsets on every machine

    return sorted(
        path 
        for path in folder.iterdir()
        # .suffix is the extension including the dot, e.g ".jpeg"; .lower() also accepts ".JPEG".
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")
    )

def collect_images(data_root):
    """Return a nested dict {split: {class: [image paths]}} covering every split and class."""
    # The / operator on a path joins path segments, the same way on Windows and Linux.

    return {
        split: {cls: list_images(data_root / split / cls) for cls in CLASSES}
        for split in SPLITS
    }

def report_counts(images):
    """Print image counts per split and class, with totals and the pneumonia: normal ratio."""
    print("--- image counts ---")
    print(f" {'split':6s} {'NORMAL':>8s} {'PNEUMONIA':>10s} {'total':>7s} {'pneu:norm':>10s}")
    grand_total = 0
    for split in SPLITS:
        normal = len(images[split]["NORMAL"])
        pneumonia = len(images[split]["PNEUMONIA"])
        total = normal + pneumonia
        grand_total += total
        # Guard so an empty folder prints "nan" instead of crashing with ZeroDivisionError
        ratio = pneumonia / normal if normal else float("nan")
        # :>8s right aligns text in 8 characters; :8d does the same for integers; :.2f is 2 decimals
        print(f" {split:6s} {normal:8d} {pneumonia:10d} {total:7d} {ratio:10.2f}")
    print(f" {'all':6s} {'':8s} {'':10s} {grand_total:7d}")

def read_header(path):
    """Return (width, height, mode) for one image, without decoding its pixels."""
    # Image.open() is lazy: it reads only the file header (size, mode) until pixel data is
    # actually requested, which keeps this fast across thousands of files.
    # 'with' closes the file afterwards. Without it thousands of handles pile up, and on 
    # Windows an open handle also locks the file.
    with Image.open(path) as img:
        width, height = img.size
        return width, height, img.mode

def describe(values):
    """Return (min, median, max) of a list of numbers."""
    return min(values), median(values), max(values)

def report_properties(images):
    """Print size, aspect-ratio and colour-mode statistics per class, across all splits.
    
    Reported per class because a systematic size or format difference between NORMAL and PNEUMONIA
    is a shortcut a model could learn instead of the pathology.
    """
    print("\n --- image properties (all splits) ---")
    for cls in CLASSES:
        # A nested comprehension reads like nested for-loops, left to right:
        # for a split in SPLITS: for path in images[splits][cls]: collect path.
        paths = [path for split in SPLITS for path in images[split][cls]]
        widths, heights, aspects = [], [], []
        # Counter is a dict that starts every missing key at 0, so modes[m] += 1 just works.
        modes = Counter()
        # tqdm wraps any iterable and draws a progress bar; leave=False clears it when done.
        for path in tqdm(paths, desc=cls, leave=False):
            width, height, mode = read_header(path)
            widths.append(width)
            heights.append(height)
            aspects.append(width / height)
            modes[mode] += 1

        w, h, a = describe(widths), describe(heights), describe(aspects)
        print(f" {cls} ({len(paths)} images)")
        print(f" width min {w[0]:5d} median {w[1]:7.0f} max {w[2]:5d}")
        print(f" height min {h[0]:5d} median {h[1]:7.0f} max {h[2]:5d}")
        print(f" aspect min {a[0]:5.2f} median {a[1]:7.2f} max {a[2]:5.2f} (width / height)")
        # most_common() returns (item, count) pairs, most frequent first.
        print(f" modes {dict(modes.most_common())}")

def patient_key(cls, path):
    """Return a candidate patient ID parsed from the filename, or None if no pattern matches.

    PNEUMONIA: bacterial and viral cases are numbered in separate sequences, so the key is the
    number plus the infection type. NORMAL: whether <A> in IM-<A>-<B> identifies a patient is
    what this survey checks, so the NORMAL2- prefix stays in the key to avoid merging two naming
    schemes.
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

def report_filenames(images):
    """Print filename-pattern and candidate-patient statistics per class.
    
    Evidence for patient-level splitting: how many image each candidate patient has, and wheter the same 
    candidate appears in more than one official split.
    """

    print("\n--- filenames and candidate patients ---")
    for cls in CLASSES:
        images_per_patient = Counter()
        splits_per_patient = {}
        unmatched = []
        for split in SPLITS:
            for path in images[split][cls]:
                key = patient_key(cls, path)
                if key is None:
                    unmatched.append(f"{split}/{path.name}")
                    continue
                images_per_patient[key] += 1
                # setdefault() returns the existing value for a key, inserting
                # the default first. if the key is missing - so each patient 
                # collects the set of splits it appears in.
                splits_per_patient.setdefault(key, set()).add(split)

        # A counter over the counts: {1: patients with 1 image, 2: patients with 2 images,...}
        distribution = Counter(images_per_patient.values())
        in_several_splits = sum(1 for splits in splits_per_patient.values() if len(splits) > 1)

        print(f" {cls}: {sum(images_per_patient.values())} matched, {len(unmatched)} unmatched")
        print(f" candidate patients: {len(images_per_patient)}")
        print(f" images per patient {dict(sorted(distribution.items()))}")
        print(f" patients in >1 split {in_several_splits}")
        if cls == "PNEUMONIA":
            all_paths = [path for split in SPLITS for path in images[split][cls]]
            # True counts as 1 when summed, so sum() over booleans counts the matches.
            bacteria = sum("bacteria" in path.name.lower() for path in all_paths)
            virus = sum("virus" in path.name.lower() for path in all_paths)
            print(f" bacteria / virus {bacteria} / {virus}")
        if unmatched:
            print(f" first unmatched {unmatched[:5]}")

def report_cross_split(images):
    """Print how PNEUMONIA person numbers, ignoring infection type, spread across official splits.

    Diagnostic for the patient key: if numbers shared across splits never agree on infection
    type, and many numbers carry both types within one split, then bacterial and viral cases are
    numbered in separate sequences and person<N> alone is not a patient ID.
    """

    print("\n--- PNEUMONIA person numbers across splits (infection type ignored) ---")
    # patient -> {split: set of infection types seen for that patient in that split}
    types_by_split = {}
    for split in SPLITS:
        for path in images[split]["PNEUMONIA"]:
            match = PNEUMONIA_NAME.fullmatch(path.stem)
            if match is None:
                continue
            patient = f"person{match.group(1)}"
            infection = match.group(2).lower()
            types_by_split.setdefault(patient, {}).setdefault(split, set()).add(infection)

    shared = {patient: splits for patient, splits in types_by_split.items() if len(splits) > 1}
    # sorted() turns {"train", "test"} into ("test", "train") so each combination counts once.
    combinations = Counter(tuple(sorted(splits)) for splits in shared.values())
    # *splits.values() unpacks the per-split sets as separate arguments; set().union() merges them.
    consistent = sum(1 for splits in shared.values() if len(set().union(*splits.values())) == 1)
    # If one number carries both types *within the same split*, the two types cannot be one patient's
    # sequence - strong evidence that bacteria and virus are numbered seperately.
    mixed_within_split = sum(
        1 for splits in types_by_split.values() if any(len(types) > 1 for types in splits.values())
    )
    print(f" shared IDs {len(shared)}")
    print(f" split combinations {dict(combinations)}")
    print(f" same infection type {consistent} of {len(shared)}")
    print(f" both types in one split {mixed_within_split}")

def parse_args():
    """Read command-line options. Only the dataset location, so the script runs anywhere."""
    # __doc__ is this module's docstring; its first line doubles as the --help description.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root",
        # type=Path converts the command-line string into a Path object automatically.
        type=Path,
        default=Path("data/raw/chest_xray"),
        help="folder containing train/, val/ and test/",
    )
    # argparse turns the dash in --data-root into an underscore: the value is args.data_root.
    return parser.parse_args()

def main():
    """Run the survey and print the report."""
    args = parse_args()
    # raise SystemExit(message) prints the message and exits with an error code - a clear
    # failure instead of a confusing traceback from deep inside iterdir().
    if not args.data_root.is_dir():
        raise SystemExit(f"Data root not found: {args.data_root.resolve()}")

    images = collect_images(args.data_root)
    report_counts(images)
    report_properties(images)
    report_filenames(images)
    report_cross_split(images)


if __name__ == "__main__":
    main()