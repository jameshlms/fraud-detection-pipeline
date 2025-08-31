from pandas import DataFrame, Series, read_csv
from sklearn.model_selection import train_test_split


def main(**kwargs) -> None:
    df = read_csv(r"../data/raw/creditcard.csv")

    X: DataFrame = df.drop(columns=["Class"])
    y: Series = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=kwargs.get("test_size", 0.2),
        random_state=kwargs.get("random_state", 42),
        stratify=y,
    )

    train_df = DataFrame(X_train)
    train_df["Class"] = y_train.values
    test_df = DataFrame(X_test)
    test_df["Class"] = y_test.values

    train_df.to_parquet(r"../data/processed/train.parquet", index=False)
    test_df.to_parquet(r"../data/processed/test.parquet", index=False)


if __name__ == "__main__":
    main()
