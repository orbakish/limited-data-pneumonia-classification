"""Grad-CAM heatmaps (Selvaraju et al. 2017, equations 1 and 2).

A forward hook captures the activations A^k of the last convolutional layer, and a backward
hook captures the gradients of the class score with respect to them. Equation 1 averages those
gradients over space into one weight per channel; equation 2 takes the ReLU of the weighted sum
of activation maps. The coarse result is bilinearly upsampled to input resolution and overlaid
on the image.

See report/papers/summaries.md for the walkthrough of both equations.

Not implemented yet - workplan section 6.
"""
