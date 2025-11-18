from os import getenv

from dotenv import load_dotenv
from numpy import cos, pi, sin
from pandas import DataFrame, Series, read_csv
from sklearn.model_selection import train_test_split as tts

load_dotenv()

RANDOM_STATE: int = int(getenv("RANDOM_STATE", 0))
TRAINING_FRACTION: float = float(getenv("TRAINING_FRACTION", 0.75))
RAW_DATA_PATH: str = getenv("RAW_DATA_PATH", "./data/raw/creditcard.csv")
TRAIN_DATA_PATH: str = getenv("TRAIN_DATA_PATH", "./data/processed/train.parquet")
TEST_DATA_PATH: str = getenv("TEST_DATA_PATH", "./data/processed/test.parquet")


def main(*args, **kwargs) -> None:
    df: DataFrame = read_csv(RAW_DATA_PATH)

    cyclic_features = DataFrame(
        {
            "hour_sin": sin(2 * pi * df["Time"] / (60 * 60 * 24)),
            "hour_cos": cos(2 * pi * df["Time"] / (60 * 60 * 24)),
        }
    )

    df: DataFrame = df.join(cyclic_features)

    X: DataFrame = df.drop(columns=["Class"])
    y: Series = df["Class"]

    train_test: tuple[DataFrame, DataFrame, Series, Series] = tts(  # type: ignore
        X,
        y,
        train_size=kwargs.get("train_size", TRAINING_FRACTION),
        random_state=kwargs.get("random_state", RANDOM_STATE),
        stratify=y,
    )

    train_df = DataFrame(train_test[0])
    train_df["Class"] = train_test[2].values
    test_df = DataFrame(train_test[1])
    test_df["Class"] = train_test[3].values

    train_df.to_parquet(TRAIN_DATA_PATH, index=False)
    test_df.to_parquet(TEST_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
