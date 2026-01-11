import pandas as pd
from datasets import Dataset
from typing import Tuple

def load_csv_dataset(
    train_path: str,
    eval_path: str,
    text_column: str,
    label_column: str,
) -> Tuple[Dataset, Dataset]:
    """
    Load train and eval datasets from CSV files.
    """

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    for df, name in [(train_df, "train"), (eval_df, "eval")]:
        if text_column not in df.columns:
            raise ValueError(f"Missing '{text_column}' column in {name} dataset")
        if label_column not in df.columns:
            raise ValueError(f"Missing '{label_column}' column in {name} dataset")

    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)

    return train_ds, eval_ds
