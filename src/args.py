class Args:
    def __init__(self, args) -> None:
        self.train_csv_path = args.train_csv_path
        self.test_csv_path = args.test_csv_path
        self.dst_dir = args.dst_dir
        self.target_column = args.target_column
        self.seed = args.seed
