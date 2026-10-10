"""Metrics derived from held-out predictions."""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score


def metric_summary(y_true, y_pred, labels):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
            "report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)}
