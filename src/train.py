"""Architecture-agnostic training loop.

Takes a config and a DataLoader and knows nothing about which of the three models it was
handed. Saves checkpoints to Google Drive in a form src/gradcam.py can later load and hook
into, and records the config, the seed and the library versions alongside each run, so no
reported number is unreproducible.

Not implemented yet - workplan section 3.
"""
