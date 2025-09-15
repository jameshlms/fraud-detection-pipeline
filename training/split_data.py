from os import getenv

from dotenv import load_dotenv
from pandas import DataFrame, Series, read_csv
from sklearn.model_selection import train_test_split

load_dotenv()
RANDOM_STATE = int(getenv("RANDOM_STATE", 0))
TRAINING_FRACTION = float(getenv("TRAINING_FRACTION", 0.75))
RAW_DATA_PATH = getenv("RAW_DATA_PATH", "./data/raw/creditcard.csv")
TRAIN_DATA_PATH = getenv("TRAIN_DATA_PATH", "./data/processed/train.parquet")
TEST_DATA_PATH = getenv("TEST_DATA_PATH", "./data/processed/test.parquet")


def main(*args, **kwargs) -> None:
    df = read_csv(RAW_DATA_PATH)

    X: DataFrame = df.drop(columns=["Class"])
    y: Series = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=kwargs.get("train_size", TRAINING_FRACTION),
        random_state=kwargs.get("random_state", RANDOM_STATE),
        stratify=y,
    )

    train_df = DataFrame(X_train)
    train_df["Class"] = y_train.values
    test_df = DataFrame(X_test)
    test_df["Class"] = y_test.values

    train_df.to_parquet(TRAIN_DATA_PATH, index=False)
    test_df.to_parquet(TEST_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
