import pandas as pd
import numpy as np
import json
from data_prep import load_and_split_data  # Custom function to split your dataset

def validate_results(test_df, params):
    """
    Validates the optimized parameters on a test dataset.

    Args:
        test_df (pd.DataFrame): The test dataset for validation.
        params (dict): The optimized parameters to validate.

    Returns:
        float: The weighted mean squared error on the test set.
    """
    prox_epsilon = params['prox_epsilon']
    prox_weight = params['prox_weight']
    gaze_weight = params['gaze_weight']

    # Calculate distances and weights
    distances = np.abs(test_df['prox_epsilon'] - prox_epsilon) + \
                np.abs(test_df['prox_weight'] - prox_weight) + \
                np.abs(test_df['gaze_weight'] - gaze_weight)
    weights = 1 / (distances + 1e-6)  # Add small epsilon to avoid division by zero

    # Compute weighted mean squared error
    weighted_mse = np.sum(weights * ((test_df['eng_1'] - test_df['grace_eng']) ** 2)) / np.sum(weights)
    return weighted_mse

def main():
    """
    Main function to validate the optimized parameters.
    """
    # Load the test dataset
    _, test_df = load_and_split_data('dataset.csv')  # Use only test data

    # Load the best parameters from the results file
    with open('results.json', 'r') as f:
        results = json.load(f)
    best_params = results['best_params']

    # Validate and print the results
    test_mse = validate_results(test_df, best_params)
    print(f"Validation Weighted MSE: {test_mse:.4f}")

if __name__ == "__main__":
    main()
