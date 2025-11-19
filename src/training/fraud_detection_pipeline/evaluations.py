from dataclasses import dataclass
from typing import Any, Union

from sklearn.base import BaseEstimator


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
