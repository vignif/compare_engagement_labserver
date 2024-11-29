import pandas as pd
from sklearn.model_selection import train_test_split
import json

def load_and_split_data(file_path):
    """
    Load the dataset from a CSV file and split it into training and testing sets.

    Args:
        file_path (str): Path to the CSV dataset file.

    Returns:
        tuple: Training and testing datasets (pd.DataFrame, pd.DataFrame).
    """
    df = pd.read_csv(file_path, sep=';')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)
    return train_df, test_df

def save_splits(train_df, test_df, train_file='train_data.json', test_file='test_data.json'):
    """
    Save the training and testing datasets to JSON files.

    Args:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Testing dataset.
    """
    train_df.to_json(train_file, orient='records', lines=True)
    test_df.to_json(test_file, orient='records', lines=True)

if __name__ == "__main__":
    train_df, test_df = load_and_split_data('dataset.csv')
    save_splits(train_df, test_df)
