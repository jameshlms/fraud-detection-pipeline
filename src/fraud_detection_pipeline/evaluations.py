from typing import Any, Callable, Union

from attr import dataclass
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def get_scores(
    *args: Callable[[ArrayLike, ArrayLike], Union[float, ArrayLike]], y_true, y_pred
) -> dict[str, Union[float, ArrayLike]]:
    scorers = args or [
        accuracy_score,
        recall_score,
        precision_score,
        f1_score,
        average_precision_score,
    ]
    return {
        metric.__name__.rstrip("_score"): metric(y_true, y_pred) for metric in scorers
    }


@dataclass(frozen=True)
class EvaluationEntry:
    name: str
    key: float
    accuracy: float
    estimator: BaseEstimator
    params: dict[str, Any]

    def __str__(self) -> str:
        return f"{self.name} (EvaluationEntry)\n- key-metric: {self.key:.4f}\n- accuracy: {self.accuracy:.4f}"


class EvaluationResults(list[EvaluationEntry]):
    def report_best(
        self,
        n: int = -1,
    ) -> Union[list[EvaluationEntry], EvaluationEntry]:
        k = min(n, len(self))
        if k <= 0 and k != -1:
            raise ValueError("k must be at least 1 or -1 for all entries")

        sorted_entries = sorted(self, key=lambda entry: entry.key, reverse=True)
        if k == 1:
            return sorted_entries[0]
        return sorted_entries[: (len(self) if k == -1 else k)]


if __name__ == "__main__":
    ...
