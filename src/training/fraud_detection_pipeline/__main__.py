from os import getenv

from dotenv import load_dotenv
from pandas import concat
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_detection_pipeline.data_loading import load_split_csv
from fraud_detection_pipeline.exporting import export_onnx, write_columns_json
from fraud_detection_pipeline.training import train
from fraud_detection_pipeline.transformations import add_cyclical_features
from fraud_detection_pipeline.utils.constants import (
    CURRENT_DIR,
    MODEL_DIRECTORY,
    RAW_DIRECTORY,
)

load_dotenv()
RANDOM_STATE = int(getenv("RANDOM_STATE", 0))
USE_ALL_CPU_CORES = bool(getenv("USE_ALL_CPUS_CORES", 0))


def main():
    X_train, X_test, y_train, y_test = load_split_csv(
        path=CURRENT_DIR / RAW_DIRECTORY / "creditcard.csv",
        target="is_fraud",
        test_size=0.3,
        random_state=RANDOM_STATE,
    )

    concat([X_train, y_train], axis=1).to_csv(
        r"../data/processed/traindata.csv", index=False
    )
    concat([X_test, y_test], axis=1).to_csv(
        r"../data/processed/testdata.csv", index=False
    )

    add_cyclical_features(X_train, "time_elapsed")

    X_train = X_train.drop(columns=["time_elapsed"])
    X_train = X_train[sorted(X_train.columns)]

    pipeline = Pipeline(
        steps=[("scaler", StandardScaler()), ("classifier", XGBClassifier())]
    )

    params = {
        "classifier__random_state": RANDOM_STATE,
        "classifier__scale_pos_weight": 552.380829015544,
        "classifier__objective": "binary:logistic",
        "classifier__subsample": 0.8,
        "classifier__n_jobs": 12,
        "classifier__alpha": 0,
        "classifier__eta": 0.01,
        "classifier__learning_rate": 0.2,
        "classifier__n_estimators": 400,
        "classifier__max_leaves": 9,
        "classifier__eval_metric": "auc",
    }

    model = train(X_train.values, y_train.values, pipeline, params)

    export_onnx(
        model,
        "fraud_detection_xgbclassifier",
        out_file=CURRENT_DIR / MODEL_DIRECTORY / "fraud_detection_xgbclassifier.onnx",
        input_dim=X_train.shape[1],
        input_name="features",
    )

    write_columns_json(
        feature_names=list(X_train.columns),
        target_column="is_fraud",
        out_file=CURRENT_DIR / MODEL_DIRECTORY / "columns_schema.json",
    )


if __name__ == "__main__":
    main()
