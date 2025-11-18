from pandas import read_csv, read_parquet
from pandas import DataFrame, Series
from pathlib import Path
from typing import Union


def _tts(
    df: DataFrame, target: str, test_size: float = 0.3, random_state: int = 0
) -> tuple[DataFrame, DataFrame, Series, Series]:
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    X = df.drop(columns=[target])
    y = df[target]

    split_index = int(len(df) * (1 - test_size))
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def tts_csv(
    path: Union[str, Path], target: str, test_size: float = 0.3, random_state: int = 0
) -> tuple[DataFrame, DataFrame, Series, Series]:
    return _tts(read_csv(path), target, test_size, random_state)


def tts_parquet(
    path: Union[str, Path], target: str, test_size: float = 0.3, random_state: int = 0
) -> tuple[DataFrame, DataFrame, Series, Series]:
    return _tts(read_parquet(path), target, test_size, random_state)
