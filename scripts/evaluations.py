"""
evaluations.py
~~~~~~~~~~~~~~

Metrics for the trained classifiers. Implemented with plain torch so the
project does not need scikit-learn just to print a confusion matrix.

Everything here works on integer label tensors:

    cm = confusion_matrix(targets, preds, len(classes))
    print(format_report(cm, classes))
"""

import torch


def confusion_matrix(targets, preds, num_classes):
    """
    Rows = true label, columns = predicted label.

    cm[i][j] = number of samples with true class i predicted as class j.
    """
    targets = torch.as_tensor(targets).flatten().long()
    preds = torch.as_tensor(preds).flatten().long()
    if targets.numel() != preds.numel():
        raise ValueError("targets and preds must have the same length")

    flat = targets * num_classes + preds
    cm = torch.bincount(flat, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def accuracy_from_cm(cm):
    total = cm.sum().item()
    return cm.diagonal().sum().item() / total if total else 0.0


def per_class_metrics(cm):
    """
    Precision / recall / F1 per class, computed straight from the matrix.

    precision = TP / (TP + FP)   column sum is who we *predicted* as that class
    recall    = TP / (TP + FN)   row sum is who actually *is* that class
    """
    cm = cm.float()
    tp = cm.diagonal()
    predicted = cm.sum(dim=0)   # column totals
    actual = cm.sum(dim=1)      # row totals

    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    recall = torch.where(actual > 0, tp / actual, torch.zeros_like(tp))

    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(denom))

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": actual.long().tolist(),
    }


def macro_f1(cm):
    f1 = per_class_metrics(cm)["f1"]
    return sum(f1) / len(f1) if f1 else 0.0


def balanced_accuracy(cm):
    """Mean recall across classes -- ignores how imbalanced the classes are."""
    recall = per_class_metrics(cm)["recall"]
    return sum(recall) / len(recall) if recall else 0.0


def format_report(cm, class_names):
    """A text block shaped like sklearn's classification_report."""
    metrics = per_class_metrics(cm)
    total = cm.sum().item()
    width = max([len(c) for c in class_names] + [7])

    lines = [
        f"{'class':<{width}} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>9}",
        "-" * (width + 40),
    ]
    for i, name in enumerate(class_names):
        lines.append(
            f"{name:<{width}} {metrics['precision'][i]:>9.3f} {metrics['recall'][i]:>9.3f} "
            f"{metrics['f1'][i]:>9.3f} {metrics['support'][i]:>9d}"
        )

    lines.append("-" * (width + 40))
    lines.append(
        f"{'accuracy':<{width}} {'':>9} {'':>9} "
        f"{accuracy_from_cm(cm):>9.3f} {total:>9d}"
    )
    lines.append(
        f"{'macro avg':<{width}} {'':>21} {macro_f1(cm):>9.3f} {total:>9d}"
    )
    return "\n".join(lines)


def format_confusion_matrix(cm, class_names):
    """Plain-text confusion matrix, rows = true, columns = predicted."""
    width = max([len(c) for c in class_names] + [10])
    head = " " * width + "".join(f"{n[:9]:>10}" for n in class_names)
    lines = [head + "   <- predicted"]

    for i, name in enumerate(class_names):
        row = "".join(f"{cm[i, j].item():>10d}" for j in range(len(class_names)))
        lines.append(f"{name:<{width}}" + row)
    return "\n".join(lines)
