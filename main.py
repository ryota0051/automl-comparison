import argparse
import os
import random
from pathlib import Path

import autokeras as ak
import numpy as np
import tensorflow as tf
from autogluon.tabular import TabularPredictor
from flaml import AutoML

from src.args import Args
from src.dataset.csv_handle import load_train_test_csv
from src.metrics import MetricsHolder
from src.plots import plot_y_true_vs_y_pred


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv_path", default="./data/train.csv")
    parser.add_argument("--test_csv_path", default="./data/test.csv")
    parser.add_argument("--dst_dir", default="./dst")
    parser.add_argument("--target_column", default="MEDV")
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def main():
    args = Args(get_args())
    metrics_holder = MetricsHolder()
    train_df, test_df = load_train_test_csv(args.train_csv_path, args.test_csv_path)
    X_train, X_test, y_train, y_test = (
        train_df.drop(columns=[args.target_column]),
        test_df.drop(columns=[args.target_column]),
        train_df[args.target_column],
        test_df[args.target_column],
    )
    # AutoGluonによる学習
    auto_gluon_dst = Path(args.dst_dir) / "AutoGluon"
    predictor = TabularPredictor(
        label="MEDV",
        problem_type="regression",
        path=str(auto_gluon_dst),
    ).fit(train_df)
    y_pred = predictor.predict(X_test)
    metrics_holder.add_result(y_test, y_pred.values, auto_gluon_dst.name)
    # flamlによる学習
    flaml_dst_dir = Path(args.dst_dir) / "flaml"
    flaml_dst_dir.mkdir(parents=True, exist_ok=True)
    flaml_dst = str(Path(args.dst_dir) / "flaml" / "logs.log")
    automl = AutoML()
    automl_settings = {
        "metric": "r2",
        "task": "regression",
        "log_file_name": flaml_dst,
        "seed": args.seed,
    }
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    y_pred = automl.predict(X_test)
    metrics_holder.add_result(y_test, y_pred, flaml_dst_dir.name)
    # auto kerasによる学習
    fix_keras_seeds(args.seed)
    auto_keras_dst_dir = Path(args.dst_dir) / "auto_keras"
    reg = ak.StructuredDataRegressor(
        max_trials=3, overwrite=True, directory=str(auto_keras_dst_dir)
    )
    reg.fit(X_train, y_train, epochs=500)
    y_pred = reg.predict(X_test)
    metrics_holder.add_result(y_test, y_pred.reshape(-1), auto_keras_dst_dir.name)
    # 結果の保存とプロット
    print("log result...")
    metrics_holder.dump_result2json(os.path.join(args.dst_dir, "result.json"))
    result_dict = metrics_holder.get_result()
    for auto_ml_name in result_dict.keys():
        elements = result_dict[auto_ml_name]
        print(auto_ml_name)
        print(f"rmse: {elements['rmse']}, mae: {elements['mae']}")
        y_pred = elements["pred"]
        plot_y_true_vs_y_pred(
            y_test.values,
            y_pred,
            Path(args.dst_dir) / auto_ml_name / "regression.png",
            title=auto_ml_name,
        )


def fix_keras_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == "__main__":
    main()
