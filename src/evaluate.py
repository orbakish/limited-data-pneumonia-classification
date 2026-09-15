"""Classification metrics for a trained model, shared by validation during training and the test run.

train.py scores the validation set with these functions every epoch, and the final test
evaluation uses the same ones, so validation and test numbers can never be computed
differently. PNEUMONIA (label 1) is the positive class.

Implemented so far: predict and scores. Still to come (workplan section 3): precision, recall,
F1, confidence intervals and the confusion matrix, written to results/metrics/E##_metrics.json
in the schema results/README.md defines.

    from src.evaluate import predict, scores
    logits, labels = predict(model, val_loader, device)
    print(scores(logits, labels))
"""

import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def predict(model, loader, device):
    """Run the model over every batch of a loader and return (logits, labels) as CPU tensors."""
    model.eval()
    all_logits, all_labels = [], []
    # no_grad skips recording operations for backpropagation - less memory, and nothing here trains.
    with torch.no_grad():
        for images, labels in loader:
            all_logits.append(model(images.to(device)).cpu())
            all_labels.append(labels)
    # torch.cat joins the per-batch tensors end to end into one tensor per quantity.
    return torch.cat(all_logits), torch.cat(all_labels)


def scores(logits, labels):
    """Return ROC-AUC and balanced accuracy for two-logit outputs, as a dict of floats.

    ROC-AUC uses the PNEUMONIA probability; balanced accuracy uses the predicted class, which for
    two logits is the larger one - the same as a 0.5 threshold on the probability.
    """
    # softmax turns each row of logits into probabilities summing to 1; column 1 is PNEUMONIA.
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    predictions = logits.argmax(dim=1)
    return {
        "roc_auc": float(roc_auc_score(labels.numpy(), probabilities.numpy())),
        "balanced_accuracy": float(balanced_accuracy_score(labels.numpy(), predictions.numpy())),
    }
