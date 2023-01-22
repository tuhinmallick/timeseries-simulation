import pandas as pd
import numpy as np


def remove_features_with_na(df, threshold=65, trailing_value=5):
    """
        Function to remove the missing features based on a % threshold. Also remove any possible trailing missing values (current rows that have missing data).

    Args:
        df (pandas.DataFrame): DataFrame with raw data. Index should be "Date".
        threshold (int, optional): Percentage threshold that is used to filter the dataframe. Defaults to 65.
        trailing_values (int, optional): Remove any trailing rows of missing values. Defaults to 5.

    Returns:
        pandas.DataFrame: Returns a fildered df with features that have more less then the 1-threshold on missing data.
    """

    # Calculate the the percentage of data missing.
    missing_values = (1 - df.isna().sum() / len(df)) * 100

    # Filter by threshold
    missing_filtered = missing_values[
        missing_values.sort_values() > threshold
    ].sort_values(ascending=True)

    # Only keep remaining cols.
    df_filtered = df[missing_filtered.index.tolist()]
    # Remove trailing missing data
    if trailing_value != 0:
        df_filtered = df_filtered.iloc[:-trailing_value]

    return df_filtered
