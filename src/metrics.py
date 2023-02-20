import json

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class MetricsHolder:
    def __init__(self) -> None:
        self.metrics_dict = {}

    def add_result(self, y_true: np.ndarray, y_pred: np.ndarray, result_name: str):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        self.metrics_dict[result_name] = {
            "mae": float(mae),
            "mse": float(mse),
            "pred": y_pred,
        }

    def dump_result2json(self, dst: str):
        with open(dst, "w") as f:
            json.dump(self.metrics_dict, f, indent=2, default=json_serial)

    def get_result(self):
        return self.metrics_dict


def json_serial(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {obj} not serializable")
