import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def plot_y_true_vs_y_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dst: str,
    title: str = None,
    xlim=(5, 55),
    ylim=(5, 55),
):
    """X軸: 目的変数, y軸: 予測結果のグラフをプロット
    Args:
        y_true: 目的変数
        y_pred: 予測結果
        dst: 保存先
        title: プロット結果のタイトル
    """
    if len(y_true.shape) != 2:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(y_true, y_pred)
    plt.scatter(y_true, y_pred, color="b", s=2)
    plt.plot(y_true, lr.predict(y_true), color="r")
    plt.xlabel("target")
    plt.ylabel("pred")
    if title:
        plt.title(title)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig(dst)
    plt.clf()
    plt.cla()
    plt.close()
