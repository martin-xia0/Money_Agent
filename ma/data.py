from __future__ import division, absolute_import, print_function
import numpy as np
import pandas as pd


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data


def data_split(df, train_ratio):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    df = df.sort_values(["nanoTime"], ignore_index=True)
    df = df.drop("Unnamed: 0", axis=1)
    split_row = int(len(df)*train_ratio)
    train_df = df[:split_row]
    test_df = df[split_row:]
    train_df.index = train_df["nanoTime"].factorize()[0]
    test_df.index = test_df["nanoTime"].factorize()[0]
    return train_df, test_df