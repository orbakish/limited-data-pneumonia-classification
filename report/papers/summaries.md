# Paper Summaries

Working summaries of the two papers this project builds on, written for what actually
matters to *implementing* the project — not generic abstract rewrites. See
[report/references.md](../references.md) for the formal citations.

---

## 1. Kermany et al. (2018) — *Cell* 172(5), 1122–1131

**"Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"**
DOI: [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)

### What the paper is about

The paper's main subject is actually retinal OCT (optical coherence tomography) image
classification — diagnosing four eye conditions (choroidal neovascularization, diabetic
macular edema, drusen, normal). The chest X-ray pneumonia task — the one our project's
dataset comes from — is a *secondary validation experiment* the authors ran to show their
transfer-learning method generalizes beyond OCT to a completely different imaging modality.
This matters context-wise: the pneumonia dataset wasn't built as the primary research
subject, it was built to prove a point about transfer learning's generality.

### Method: transfer learning recipe

The core technique, applied identically to both the OCT and X-ray experiments:

1. Start from an **Inception V3** CNN pretrained on ImageNet (1000 general object classes).
2. **Freeze** all convolutional layers — these become a fixed feature extractor. The
   activations these frozen layers produce for each image are called "bottlenecks," and are
   computed once and cached, since they never change during retraining.
3. **Replace and retrain only the final classification layer** (randomly initialized) on the
   bottleneck features, using the project's own classes (4 for OCT, or normal/pneumonia and
   bacterial/viral for X-rays).
4. Training: stochastic gradient descent, Adam optimizer, learning rate 0.001, batch size
   1,000, run for 10,000 steps (≈100 epochs).

**Important negative result**: the authors also tried *fine-tuning* — unfreezing the
convolutional layers and updating them via backprop on the medical images — and found it
**decreased** performance due to overfitting on the limited medical dataset. This is directly
relevant to our project: it's a documented precedent that more aggressive adaptation of a
pretrained network can hurt rather than help when data is scarce, which is exactly the
regime our 20%/50%/100% data-size axis is probing.

### Pneumonia results (the numbers behind our dataset)

- 5,232 training images (3,883 pneumonia: 2,538 bacterial + 1,345 viral; 1,349 normal) from
  5,856 patients.
- Test set: 234 normal + 390 pneumonia (242 bacterial + 148 viral) from 624 patients.
- **Pneumonia vs. normal**: 92.8% accuracy, 93.2% sensitivity, 90.1% specificity, AUC 96.8%.
- **Bacterial vs. viral** (a finer-grained task than ours, since our project is just
  pneumonia-vs-normal): 90.7% accuracy, 88.6% sensitivity, 90.9% specificity, AUC 94.0%.

### The "limited data still works" precedent

On the OCT task, the authors also trained a **"limited model"** using only 1,000 images per
class (instead of the full ~27,000/class average) and compared it to the full model:

| | Accuracy | Sensitivity | Specificity | Weighted error |
|---|---|---|---|---|
| Full model | 96.6% | 97.8% | 97.4% | 6.6% |
| Limited model (1,000/class) | 93.4% | 96.6% | 94.0% | 12.7% |

This is essentially the OCT-domain precedent for our project's core question: transfer
learning holds up reasonably well under data scarcity, but does degrade — and our project is
asking the same question more systematically (3 data sizes × 3 augmentation regimes × 3
architectures) and adding the explanation-stability angle they never measured.

### Evaluation formulas used

The paper reports standard classification metrics, defined the standard way:

- **Accuracy** = (correctly classified images) / (total test images)
- **Sensitivity** (recall on the positive/"urgent" class) = (correctly identified positive
  cases) / (total actual positive cases)
- **Specificity** = (correctly identified negative cases) / (total actual negative cases)
- **ROC curve / AUC**: sensitivity (true positive rate) plotted against 1 − specificity
  (false positive rate) as the classification threshold varies; AUC is the probability the
  model ranks a random positive example above a random negative one.

They also introduce a **weighted error score** for the OCT referral task, to reflect that a
missed urgent case is worse than a false alarm: an "urgent referral" (CNV/DME) misclassified
as *normal* gets a penalty of 4, misclassified as *drusen* gets a penalty of 2, and any other
wrong answer gets a penalty of 1. This isn't something our project needs to replicate (our
task is a plain binary pneumonia/normal classification evaluated with standard classification
metrics including accuracy, precision, recall, specificity, F1, balanced accuracy, and ROC-AUC),
but it's a useful pattern to be aware of if the report ever wants to argue that not all
misclassifications are equally costly in a medical setting.

### Interpretability method: occlusion testing (not Grad-CAM)

To show their model wasn't a "black box," the authors used **occlusion testing**: slide a
blank 20×20 pixel patch across every position in the image, and at each position, note how
much the model's predicted probability for the correct class drops when that patch is
blanked out. The position causing the biggest drop is inferred to be the most important
region for the decision. This produces a coarse heatmap, but it is *not* Grad-CAM — it
requires one forward pass per patch position (expensive), whereas Grad-CAM needs only one
forward + one backward pass per image.

**Why this matters for our project**: our project explicitly replaces/upgrades this
occlusion-based interpretability approach with Grad-CAM, and goes a step further by measuring
whether the resulting explanations are *stable* under small perturbations — something neither
paper does. That's the actual novelty angle for our report's introduction.

---

## 2. Selvaraju et al. (2017) — ICCV, "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization"

DOI: [10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74) · arXiv:1610.02391

### What problem it solves

CNNs are accurate but opaque. Prior visualization techniques fell into two camps, both with a
flaw:
- **Pixel-space gradient methods** (Guided Backpropagation, Deconvolution): high-resolution
  and detailed, but **not class-discriminative** — the visualization for "cat" looks almost
  identical to the visualization for "dog" on the same image, because they highlight *any*
  edge/texture the network reacted to, not what's specific to the predicted class.
  - **Class Activation Mapping / CAM** (Zhou et al.): class-discriminative, but only works on
  a narrow architecture family (must end in global-average-pooling directly into a softmax
  layer) — using it on an arbitrary architecture requires modifying and retraining the model.

Grad-CAM's contribution: get CAM's class-discriminative localization on **any** CNN, with no
architecture changes and no retraining — including exactly the kind of architectures this
project uses (ResNet18) and adjacent modern families (MobileNet, EfficientNet), none of which
were designed with CAM's global-average-pool-into-softmax constraint in mind.

### The algorithm — the part you'll actually implement

Given an image and a target class `c`, Grad-CAM produces a coarse heatmap over the spatial
locations of a chosen convolutional layer (in practice, the *last* convolutional layer,
because it has the best balance of high-level class semantics and remaining spatial
resolution — earlier layers have more spatial detail but less semantic meaning, so
localization gets progressively worse the earlier in the network you go, per Fig. 13/14 in
the paper).

**Step 1 — Neuron importance weights (Eq. 1):**

```
α^c_k = (1/Z) · Σ_i Σ_j ( ∂y^c / ∂A^k_ij )
```

Read this as: for feature-map channel `k` (out of however many channels the chosen conv
layer has), take the gradient of the class score `y^c` (the raw score *before* softmax, for
the class you're explaining) with respect to every spatial position `(i, j)` of that
channel's activation map `A^k`, then average all of those gradient values together (`Z` is
the number of pixels in the map, i.e. width × height — dividing by it is exactly a mean, aka
"global average pooling" of the gradients). The result, `α^c_k`, is a single number per
channel: how much, on average, increasing channel `k`'s activation would increase the score
for class `c`. This is obtained with one backward pass — no extra forward passes needed,
unlike occlusion testing.

**Step 2 — Weighted combination + ReLU (Eq. 2):**

```
L^c_Grad-CAM = ReLU( Σ_k α^c_k · A^k )
```

Take every channel's activation map `A^k`, scale it by its importance weight `α^c_k` from
step 1, and sum across all channels. This weighted sum is a single-channel heatmap the same
spatial size as the conv layer (e.g. 7×7). Then apply **ReLU** — clip negative values to
zero. The ReLU matters a lot: the authors' ablation study (Table 3) shows removing it
increases localization error by ~15 percentage points, because negative values in the raw sum
indicate evidence *against* the class (or for a different class), and keeping them in muddies
the map. Only positive evidence — pixels that, if intensified, would increase confidence in
this specific class — should show up.

The resulting `L^c_Grad-CAM` is coarse (matches the conv layer's spatial resolution, not the
input image's), so it's typically upsampled to the input resolution via bilinear
interpolation before being overlaid on the image or compared to another heatmap.

**Design-choice ablations worth knowing** (Table 3, Sec. B of the appendix):
- **Global-average-pooling of gradients beats global-max-pooling** for the weighting step —
  average is more robust to noise than max.
- Modifying the backward pass of ReLU during backprop (as in Guided Backprop or
  Deconvolution) makes the *class-discriminativeness* worse, even though it can slightly
  help raw localization numbers — the paper recommends using plain, unmodified gradients for
  Grad-CAM itself (any ReLU modification belongs only in the separate Guided Backprop
  branch, if you're building Guided Grad-CAM, not plain Grad-CAM).

### Guided Grad-CAM (not required by our project, but good to know)

The paper also proposes combining Grad-CAM with Guided Backpropagation for higher-resolution
visualizations: elementwise-multiply the (upsampled) Grad-CAM heatmap with a Guided
Backpropagation map of the same image. This gives a result that's both high-resolution and
class-discriminative. Our project's plan only calls for Grad-CAM itself (used for the
explanation-stability analysis with Pearson correlation), not the Guided variant, but it's worth
knowing the name in case it comes up in report/report review as "why not Guided Grad-CAM."

### Counterfactual explanations (Eq. 12, likely not needed, but relevant vocabulary)

A one-line variant: negate the gradient before pooling —

```
α^c_k = (1/Z) · Σ_i Σ_j ( −∂y^c / ∂A^k_ij )
```

— which highlights regions that, if *removed*, would make the network *more* confident in
its prediction (i.e. regions currently acting against the predicted class). Not part of our
project's plan, but useful vocabulary if it comes up.

### Why Grad-CAM is a generalization of CAM (Eqs. 3–11)

The paper proves that for the specific architectures CAM already supports (global-average-
pool straight into softmax), Grad-CAM's weights `α^c_k` reduce to exactly CAM's weights
`w^c_k`, up to a constant factor. The derivation, in short:

- CAM defines the class score as a linear combination of the globally-average-pooled feature
  maps: `Y^c = Σ_k w^c_k · F^k`, where `F^k = (1/Z) Σ_i Σ_j A^k_ij` is feature map `k`
  averaged over space, and `w^c_k` is a learned weight connecting that feature map to class
  `c` (the softmax layer's weights, essentially).
- Differentiating `Y^c` with respect to `F^k` just gives back `w^c_k` (since the relationship
  is linear) — that's the left-hand side of Eq. 6.
- Differentiating `F^k` with respect to a single activation `A^k_ij` gives `1/Z` (since `F^k`
  is a plain average) — Eq. 6's other half.
- Chaining these through algebra (Eqs. 7–11) shows `w^c_k = Σ_i Σ_j (∂Y^c / ∂A^k_ij)` — which
  is exactly Grad-CAM's `α^c_k` (Eq. 1) without the `1/Z` averaging constant (a constant that
  cancels out when you normalize the heatmap for visualization anyway).

**Why this matters practically**: it's the formal justification for why Grad-CAM is a safe
drop-in replacement for CAM-style localization on architectures CAM can't touch — it isn't a
heuristic that happens to look similar, it's provably the same computation generalized to
work via gradients instead of requiring a specific architecture. For our project (ResNet18,
MobileNetV3-Small, EfficientNet-B0 — none of which end in a bare global-average-pool→softmax
the way CAM requires), this is exactly why Grad-CAM is the right tool and CAM isn't an option
at all without modifying and retraining every model.

### Implementation notes for our 3 architectures

- Grad-CAM needs a **forward hook** (to capture the target layer's activations `A^k` during
  the forward pass) and a **backward hook** (to capture the gradients `∂y^c/∂A^k` during
  backprop) registered on the chosen convolutional layer.
- The target layer should be the **last convolutional layer before pooling/classification**
  for each architecture — the exact module name differs per architecture (e.g. in
  torchvision's ResNet18 it's the output of `layer4`; MobileNetV3-Small and EfficientNet-B0
  have their own final-feature-block names) and needs pinning down per model when we
  implement this.
- The ResNet-specific finding in the paper's appendix (Sec. G): Grad-CAM localizes well at
  the last conv layer and the last residual block, but **degrades sharply at downsampling
  layers** earlier in the network. Confirms: always hook the last conv block, not an
  arbitrary intermediate one.
