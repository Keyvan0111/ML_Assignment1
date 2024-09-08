"""
Author: Keyvan Sadeghi
Short: Common functions
"""

import pandas as pd # type: ignore
import numpy as np

def get_songs(file_name: str):
    """
    This function returns some dataset from a csv file as a matrix
    """

    data_frame = pd.read_csv(file_name)
    return data_frame

def get_songs_by_genre(data_frame, genre: str):
    """
    Args: genre (str)
    return: dataset (pd.data_frame)
    """
    label = 1 if genre == 'Pop' else 0
    filtered_data_frame = data_frame[data_frame['genre'] == label]
    return filtered_data_frame
    
def add_label(data_frame, feature: str) -> None:
    """
    Adds label to "feature" column
    If the value is Pop then label is set to 1, and 0 otherwise

    Args:
        data_frame: pd.DataFrame -  dataframe from csv file
        feature: str - Column name
    """
    data_frame[feature] = np.where(data_frame[feature] > 'Pop', 1, 0)

def data_filter(data_frame, features: list[str]):
    """
    Filter data based on features given.
    Returns new data_frame only with the given features
    """
    return data_frame[features]
