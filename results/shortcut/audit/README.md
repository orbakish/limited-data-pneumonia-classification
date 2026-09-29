# Shortcut audit (NOTES.md #15)

Written by `python -m src.shortcut --audit-sheets`. For each numbered overlay, write the main hot spot's
place in `labels.csv` (a copy of `labels_template.csv`), one of: lung field, mediastinum-heart, image edge-background, marker-text, device-wires, below the diaphragm, diffuse-other. A
second label is optional. Label overlays 1-60 first, then any after 60. Do not open `key.csv` while
labelling: it says which run, image and map each overlay is.
