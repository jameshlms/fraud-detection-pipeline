from os import getenv

import numpy as np
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_detection_pipeline.data_loading import tts_csv
from fraud_detection_pipeline.export import to_onnx_file, write_column_names_json
from fraud_detection_pipeline.utils.constants import (
    CPU_COUNT,
    CURRENT_DIR,
    MODEL_DIRECTORY,
    PROCESSED_DIRECTORY,
    SCHEMA_DIRECTORY,
    TARGET_COLUMN,
)

load_dotenv()

RANDOM_STATE = int(getenv("RANDOM_STATE", 0))
USE_ALL_CPU_CORES = bool(getenv("USE_ALL_CPUS_CORES", 0))
TEST_SPLIT_SIZE = float(getenv("TEST_SPLIT_SIZE", 0.3))


splits = tts_csv(
    CURRENT_DIR / PROCESSED_DIRECTORY / "creditcard.csv",
    TARGET_COLUMN,
    test_size=TEST_SPLIT_SIZE,
    random_state=RANDOM_STATE,
)

num_fraud = splits[3].sum()
num_genuine = splits[3].count() - num_fraud

write_column_names_json(
    column_names=splits[0].columns.tolist(),
    target_column=TARGET_COLUMN,
    out_file=CURRENT_DIR / SCHEMA_DIRECTORY / "columns.json",
)


X_train, X_test, y_train, y_test = (
    np.asarray(split).astype(np.float32) for split in splits
)

numerical = list(range(X_train.shape[1]))

preproc = ColumnTransformer(
    [
        ("numerical", StandardScaler(), numerical),
    ],
    remainder="drop",
)

xgboost = XGBClassifier(
    seed=RANDOM_STATE,
    scale_pos_weight=num_genuine / num_fraud,
    objective="binary:logistic",
    subsample=0.8,
    n_jobs=CPU_COUNT if USE_ALL_CPU_CORES else 1,
    alpha=0,
    eta=0.01,
    learning_rate=0.2,
    n_estimators=400,
    max_leaves=9,
    eval_metric="auc",
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preproc),
        ("classifier", xgboost),
    ]
)

pipeline.fit(X_train, y_train)

to_onnx_file(
    pipeline,
    name="FraudXGBClassifier",
    out_file=CURRENT_DIR / MODEL_DIRECTORY / "fraud_xgb_classifier.onnx",
    input_dim=X_train.shape[1],
)
