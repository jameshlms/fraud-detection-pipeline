import json
from pathlib import Path

from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier

update_registered_converter(
    XGBClassifier,
    "XGBoostXGBClassifier",
    calculate_linear_classifier_output_shapes,
    convert_xgboost,
    options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
)


def export_onnx(
    model: BaseEstimator,
    name: str,
    out_file: str | Path,
    input_dim: int,
    input_name: str = "input",
):
    if not str(out_file).endswith(".onnx"):
        raise ValueError("Output file must have a .onnx extension")
    onnx_model = convert_sklearn(
        model,
        name=name,
        initial_types=[(input_name, FloatTensorType([None, input_dim]))],
        target_opset={"": 12, "ai.onnx.ml": 2},
    )
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        f.write(onnx_model.SerializeToString())


def write_columns_json(
    feature_names: list[str], target_column: str, out_file: str | Path
):
    data = {
        "feature_names": feature_names,
        "target_name": target_column,
    }
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(data, f, indent=4)
