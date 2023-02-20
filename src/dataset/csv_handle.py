import pandas as pd


def load_train_test_csv(train_csv_path: str, test_csv_path: str):
    """学習用とテスト用csvをロードする
    Args:
        train_csv_path: 学習用csvパス
        test_csv_path: テスト用
    Examples:
        >>> train_csv_path = './data/train.csv'
        >>> test_csv_path = './data/test.csv'
        >>> train_df, test_df = load_train_test_csv(train_csv_path, test_csv_path)
    """
    # csvロード
    train_df, test_df = pd.read_csv(train_csv_path), pd.read_csv(test_csv_path)
    return train_df, test_df
