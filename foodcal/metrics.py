"""Classification counts with explicit support for every configured class."""

import numpy as np

from .model import validate_classes


def classification_metrics(y_true, y_pred, class_names):
    classes = validate_classes(class_names)
    truth, predicted = np.asarray(y_true), np.asarray(y_pred)
    if (truth.ndim != 1 or predicted.shape != truth.shape or not truth.size
            or truth.dtype.kind not in "iu" or predicted.dtype.kind not in "iu"
            or (truth < 0).any() or (truth >= len(classes)).any()
            or (predicted < 0).any() or (predicted >= len(classes)).any()):
        raise ValueError("Labels must be equal-length nonempty integer vectors within the class order")
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    support = matrix.sum(axis=1)
    output_count = matrix.sum(axis=0)
    correct = matrix.diagonal()
    precision = np.divide(correct, output_count, out=np.zeros(len(classes), dtype=float), where=output_count > 0)
    recall = np.divide(correct, support, out=np.zeros(len(classes), dtype=float), where=support > 0)
    f1 = np.divide(2 * correct, support + output_count,
                   out=np.zeros(len(classes), dtype=float), where=support + output_count > 0)
    return {
        "class_names": list(classes), "sample_count": int(truth.size),
        "accuracy": float(correct.sum() / truth.size), "macro_f1": float(f1.mean()),
        "macro_f1_present_classes": float(f1[support > 0].mean()),
        "per_class_recall": [float(recall[i]) if support[i] else None for i in range(len(classes))],
        "support": support.tolist(), "confusion_matrix": matrix.tolist(),
        "per_class": {name: {"support": int(support[i]), "predicted": int(output_count[i]),
                             "precision": float(precision[i]) if output_count[i] else None,
                             "recall": float(recall[i]) if support[i] else None, "f1": float(f1[i])}
                      for i, name in enumerate(classes)},
        "macro_f1_policy": "All configured classes; undefined class F1 is zero; absent-class recall is null",
    }
