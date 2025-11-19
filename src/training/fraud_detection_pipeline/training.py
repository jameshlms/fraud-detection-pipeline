from os import getenv
from typing import Any

from dotenv import load_dotenv
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

load_dotenv()

RANDOM_STATE = int(getenv("RANDOM_STATE", 0))


def train(
    X: ArrayLike,
    y: ArrayLike,
    model: BaseEstimator,
    params: dict[str, Any],
) -> BaseEstimator:
    return model.set_params(**params).fit(X, y)
