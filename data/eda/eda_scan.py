# -*- coding: utf-8 -*-
"""
READ-ONLY EDA scan for Chest X-Ray dataset.
Produces image_inventory.csv, and prints stats used to build the report.
Does NOT modify, move, rename, or delete any data file.
"""
import os
import sys
import re
import csv
import hashlib
import io
import warnings

from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
# Do NOT allow truncated images to load silently -> we want to detect them.
ImageFile.LOAD_TRUNCATED_IMAGES = False
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = ROOT
SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]

IMG_EXTS = {".jpeg", ".jpg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
SYSTEM_FILES = {".ds_store", "thumbs.db", "desktop.ini"}

def is_system_file(name):
    low = name.lower()
    if low in SYSTEM_FILES:
        return True
    if low == ".ds_store":
        return True
    if low.startswith("._"):  # macOS AppleDouble
        return True
    return False

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_patient_candidate(cls, filename):
    """Best-effort patient/group id candidate extraction.
    PNEUMONIA: 'personN_...'  -> 'person{N}'
    NORMAL: analyzed separately; return normalized group key candidate.
    Returns (candidate, method).
    """
    base = filename
    # strip extension
    stem = os.path.splitext(base)[0]
    if cls == "PNEUMONIA":
        m = re.match(r"^(person\d+)", stem, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower(), "person_regex"
        return "UNKNOWN", "no_person_prefix"
    else:  # NORMAL
        # Pattern A: IM-0115-0001  -> group IM-0115
        # Pattern B: NORMAL2-IM-0416-0001(-0001) -> group NORMAL2-IM-0416
        m = re.match(r"^(NORMAL2-IM-\d+)", stem)
        if m:
            return m.group(1), "normal2_im"
        m = re.match(r"^(IM-\d+)", stem)
        if m:
            return m.group(1), "im"
        return stem, "raw_stem"

def main():
    rows = []
    inventory_path = os.path.join(OUT_DIR, "image_inventory.csv")
    non_image_files = []   # (split, class, path, reason)
    corrupted = []         # (split, class, path, error)
    total_seen = 0

    for split in SPLITS:
        for cls in CLASSES:
            d = os.path.join(ROOT, split, cls)
            if not os.path.isdir(d):
                print(f"WARN missing dir: {d}", file=sys.stderr)
                continue
            for name in sorted(os.listdir(d)):
                path = os.path.join(d, name)
                if not os.path.isfile(path):
                    continue
                if is_system_file(name):
                    continue  # ignore system files completely
                total_seen += 1
                ext = os.path.splitext(name)[1].lower()
                try:
                    fsize = os.path.getsize(path)
                except OSError:
                    fsize = -1

                if ext not in IMG_EXTS:
                    non_image_files.append((split, cls, path, f"ext={ext}"))
                    # still record row minimally
                    rows.append({
                        "path": path, "split": split, "class": cls,
                        "filename": name, "extension": ext,
                        "width": "", "height": "", "aspect_ratio": "",
                        "mode": "", "channels": "",
                        "patient_id_candidate": "", "patient_id_method": "",
                        "file_size": fsize, "hash": "", "status": "non_image",
                    })
                    continue

                # Compute hash (also forces OneDrive download)
                try:
                    fhash = sha256_of_file(path)
                except Exception as e:
                    fhash = ""
                    corrupted.append((split, cls, path, f"hash_error:{e}"))

                width = height = None
                mode = ""
                channels = ""
                status = "ok"
                try:
                    with Image.open(path) as im:
                        width, height = im.size
                        mode = im.mode
                    # Full decode to detect truncation/corruption
                    with Image.open(path) as im2:
                        im2.load()
                    channels = len(Image.open(path).getbands())
                except Exception as e:
                    status = "corrupted"
                    corrupted.append((split, cls, path, str(e)))

                ar = ""
                if width and height:
                    ar = round(width / height, 6)

                pcand, pmethod = extract_patient_candidate(cls, name)

                rows.append({
                    "path": path, "split": split, "class": cls,
                    "filename": name, "extension": ext,
                    "width": width if width is not None else "",
                    "height": height if height is not None else "",
                    "aspect_ratio": ar,
                    "mode": mode, "channels": channels,
                    "patient_id_candidate": pcand, "patient_id_method": pmethod,
                    "file_size": fsize, "hash": fhash, "status": status,
                })
            print(f"done {split}/{cls}: cumulative rows={len(rows)}", flush=True)

    # write inventory
    fields = ["path","split","class","filename","extension","width","height",
              "aspect_ratio","mode","channels","patient_id_candidate",
              "patient_id_method","file_size","hash","status"]
    with open(inventory_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("=== SUMMARY ===")
    print("total_seen(non-system):", total_seen)
    print("rows:", len(rows))
    print("non_image_files:", len(non_image_files))
    for x in non_image_files:
        print("  NONIMG", x)
    print("corrupted:", len(corrupted))
    for x in corrupted:
        print("  CORRUPT", x)
    print("inventory written:", inventory_path)

if __name__ == "__main__":
    main()
