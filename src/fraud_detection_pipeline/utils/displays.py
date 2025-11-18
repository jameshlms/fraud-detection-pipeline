from pandas import Series
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


def proportions_values_counts(series: Series):
    proportions = series.value_counts(normalize=True).apply(lambda x: f"{x:.2%}")
    return str(proportions)


def mertrics_report(y_true, y_pred):
    report = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "average_precision": average_precision_score(y_true, y_pred),
    }
    return str(Series(report).apply(lambda x: f"{x:.4f}"))
