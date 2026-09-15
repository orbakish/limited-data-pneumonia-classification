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
import random
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from tqdm import tqdm

# The folder layout, filename patterns and file listing live in src/paths.py, shared with the
# training pipeline - so the survey measures exactly the patient keys the splits are built on.
from src.paths import CLASSES, PNEUMONIA_NAME, SPLITS, collect_images, patient_key

SEED = 42
BORDER_SAMPLE = 300 # images per class for the border measurement
DARK_THRESHOLD = 0.08 # mean intensity (0-1) below which an edge row or column counts as dark background
SHEET_SAMPLE = 16                 # images per class on each contact sheet (4 x 4 grid)
INSET_GUIDES = (0.05, 0.10)       # guide lines drawn this far in from every edge




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

def leading_below(values, threshold):
    """Return how many values at the start of the sequence are below the threshold."""
    count = 0
    for value in values:
        if value >= threshold:
            break
        count += 1
    return count

def dark_border_fractions(path):
    """Return dark-border thickness at each edge as a fraction of image size.
    
    Returns (top, bottom, left, right). A row or column counts as border while its mean intensity stays
    below DARK_THRESHOLD, scanning inwards from that edge.
    """
    with Image.open(path) as img:
        # draft() asks the JPEG decoder to decode at a reduced scale directly - far faster
        # than decoding at full size and resizing afterwards.
        img.draft("L", (512, 512))
        # convert("L") guarantees single-channel grayscale, including the 283 RGB files.
        # np.asarray turns the image into an array of pixel values; /255 rescales to 0-1.
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    # .mean(axis=1) averages across each row (one value per row); axis=0 does the same per column.
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    height, width = gray.shape

    #[::-1] reverses an array, so the same scan can works inwards from the bottom and the right edges.
    return (
        leading_below(row_means, DARK_THRESHOLD) / height,
        leading_below(row_means[::-1], DARK_THRESHOLD) / height,
        leading_below(col_means, DARK_THRESHOLD) / width,
        leading_below(col_means[::-1], DARK_THRESHOLD) / width,
    )

def report_borders(images):
    """Print dark background at each image edge, per class, on a seeded sample of images.

    Framing evidence, not a lung-margin measurement: these images have no scanner border, so the
    scan stops at the first bright column, usually an arm or shoulder. It shows how often each
    class includes empty background beside the body, which differs between the classes.
    """
    print(f"\n--- dark background at image edges (sample of {BORDER_SAMPLE} per class, fraction of size) ---")
    # random.Random(seed) is a private random generator: reproducible, and it leaves Pyhtons's
    # global random state untouched.
    rng = random.Random(SEED)
    for cls in CLASSES:
        paths = [path for split in SPLITS for path in images[split][cls]]
        sample = rng.sample(paths, min(BORDER_SAMPLE, len(paths)))
        # np.array over a list of 4-tuples builds a table: one row per image, one column per edge
        fractions = np.array([dark_border_fractions(p) for p in tqdm(sample, desc=cls, leave=False)])


        print(f" {cls}")
        for i, edge in enumerate(("top", "bottom", "left", "right")):
            # fractions[:, i] is column i: that edge's value for every sampled image.
            column = fractions[:, i]
            # np.percentile(x, 90) is the value 90% of images fall at or below.
            # (column > 0) is an array of True/False; its mean is the share of True.
            print(f" {edge:6s} median {np.median(column):.3f} "
                  f"p90 {np.percentile(column, 90):.3f} "
                  f"with any border {np.mean(column > 0):.0%}")

def load_thumbnail(path, size=256):
    """Return a grayscale thumbnail of an image as an array, keeping its true aspect ratio."""
    with Image.open(path) as img:
        img.draft("L", (size * 2, size * 2))
        img = img.convert("L")
        # thumbnail() shrinks the image in place to fit inside (size, size) without distorting it.
        img.thumbnail((size, size))
        return np.asarray(img)


def draw_sheet(paths, title, output_path, columns=4):
    """Save a grid of thumbnails, one cell per image, with dashed inset guide lines."""
    # -(-a // b) is ceiling division: // rounds down, so negating twice rounds up instead.
    rows = -(-len(paths) // columns)
    # Figure draws off-screen without opening a window, unlike pyplot, so it works the same
    # locally and on Colab.
    fig = Figure(figsize=(columns * 3, rows * 3))
    fig.suptitle(title)
    # squeeze=False keeps axes as a 2-D grid even with a single row, so .flat always works.
    axes = fig.subplots(rows, columns, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    # zip() pairs cells with images and stops at the shorter one, so spare cells stay blank.
    for ax, path in zip(axes.flat, paths):
        thumb = load_thumbnail(path)
        height, width = thumb.shape
        ax.imshow(thumb, cmap="gray")
        for inset, colour in zip(INSET_GUIDES, ("yellow", "red")):
            # Content outside a guide line is what a translation of that size could push out of frame.
            ax.axhline(inset * height, color=colour, linewidth=0.6, linestyle="--")
            ax.axhline((1 - inset) * height, color=colour, linewidth=0.6, linestyle="--")
            ax.axvline(inset * width, color=colour, linewidth=0.6, linestyle="--")
            ax.axvline((1 - inset) * width, color=colour, linewidth=0.6, linestyle="--")
        ax.set_title(path.name, fontsize=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)

def report_contact_sheets(images, output_dir):
    """Save contact sheets: a seeded random sample per class, plus each class's shape outliers.

    For judging by eye how close the lungs come to the frame edge (question #11). The sheets
    show patient images, so output_dir must stay gitignored.
    """
    print(f"\n--- contact sheets -> {output_dir} ---")
    # parents=True creates missing parent folders; exist_ok=True makes re-running harmless.
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    extremes = []

    for cls in CLASSES:
        paths = [path for split in SPLITS for path in images[split][cls]]
        sample = rng.sample(paths, min(SHEET_SAMPLE, len(paths)))
        sheet = output_dir / f"sample_{cls.lower()}.png"
        draw_sheet(sample, f"{cls}: random sample (seed {SEED}), guides at 5% yellow / 10% red", sheet)
        print(f"  saved {sheet}")

        headers = {path: read_header(path) for path in paths}
        # key= tells min()/max() what to compare. `lambda p: ...` is a one-line unnamed function;
        # read_header returned (width, height, mode), so [1] is height and [0] / [1] is aspect.
        extremes += [
            min(paths, key=lambda p: headers[p][1]),
            min(paths, key=lambda p: headers[p][0] / headers[p][1]),
            max(paths, key=lambda p: headers[p][0] / headers[p][1]),
        ]
    sheet = output_dir / "extremes.png"
    draw_sheet(extremes, "Per class: shortest, narrowest, widest (row 1 NORMAL, row 2 PNEUMONIA)",
               sheet, columns=3)
    print(f"  saved {sheet}")

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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/survey"),
        help="where contact sheets are saved (must stay gitignored)",
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
    report_borders(images)
    report_contact_sheets(images, args.output_dir)



if __name__ == "__main__":
    main()