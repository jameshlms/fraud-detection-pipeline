import os


def create_default_env(overwrite: bool = False) -> bool:
    if not overwrite and os.path.exists(".env"):
        print(".env file already exists. Use overwrite=True to overwrite.")
        return False
    with open(".env", "w") as env:
        env.writelines(
            (
                "RANDOM_STATE=39103\n",
                "TRAINING_FRACTION=0.75\n",
                "RAW_DATA_PATH=./data/raw/creditcard.csv\n",
                "TRAIN_DATA_PATH=./data/processed/train.parquet\n",
                "TEST_DATA_PATH=./data/processed/test.parquet\n",
            )
        )

    return True


if __name__ == "__main__":
    result = create_default_env()
