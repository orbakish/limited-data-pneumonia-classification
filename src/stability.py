"""Explanation stability: how much a Grad-CAM heatmap moves under a small input perturbation.

Measured two ways on a fixed class-balanced test subset, and both are reported:

* In-family - a small rotation. The perturbed heatmap is rotated back into alignment and the
  corners that rotated out of frame are masked before comparing, otherwise even a perfectly
  consistent model scores low simply because the lungs moved.
* Out-of-family - small Gaussian noise, which no augmentation regime trains on and which moves
  no pixels, so the heatmaps compare directly.

Reporting both matters: augmentation trains invariance to specific changes, so an in-family
perturbation flatters augmented models by construction. The gap between the two is itself a
finding.

Not implemented yet - workplan section 7.
"""
