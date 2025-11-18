from os import getenv
from typing import NamedTuple

from onnxruntime import InferenceSession
from dotenv import load_dotenv
from numpy import cos, pi, sin
from pandas import DataFrame, Series, read_parquet
from skl2onnx import to_onnx
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

load_dotenv()

RANDOM_STATE = int(getenv("RANDOM_STATE", 0))

transactions: DataFrame = read_parquet(r"./data/processed/creditcard.parquet")

X = transactions.drop(columns=["is_fraud"])

X_train, X_test, y_train, y_test = tts(
    transactions.drop(columns=["is_fraud"]),
    transactions["is_fraud"],
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=transactions["is_fraud"],
)

numerical = (
    transactions.select_dtypes(include=["float64", "int64", "int32"])
    .drop(columns="time")
    .columns.tolist()
)

categorical = transactions.select_dtypes(
    include=["object", "category"]
).columns.tolist()

preproc = ColumnTransformer(
    [
        (
            "numerical",
            StandardScaler(),
            numerical,
        ),
    ],
    remainder="passthrough",
)

log_base = SkPipeline(
    steps=[
        ("preprocessor", preproc),
        (
            "classifier",
            LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1500,
                class_weight="balanced",
            ),
        ),
    ]
)

log_base.fit(X_train, y_train)

infer_pipe = SkPipeline(
    steps=[
        ("preprocessor", log_base.named_steps["preprocessor"]),
        ("classifier", log_base.named_steps["classifier"]),
    ]
)

onnx_model = to_onnx(
    infer_pipe,
    X_train[:1].astype("float32"),  # type: ignore (type checking)
)

with open(r"./models/fraud_classifier.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())  # type: ignore (type checking)

sess = InferenceSession(
    "./models/fraud_classifier.onnx", providers=["CPUExecutionProvider"]
)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype("float32")})[0]
