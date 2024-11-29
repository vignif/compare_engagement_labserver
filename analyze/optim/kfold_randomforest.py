import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path, sep=';')

# Optimize parameters without KFold
def optimize_parameters(df, param_grid):
    """
    Optimize gaze_weight and prox_epsilon using RandomForestRegressor and a single train/test split.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        param_grid (dict): Parameter grid with 'gaze_weight' and 'prox_epsilon' ranges.

    Returns:
        dict: Best parameters (gaze_weight, prox_epsilon) and their MSE.
    """
    best_mse = float('inf')
    best_params = {'gaze_weight': None, 'prox_epsilon': None}

    # Split the dataset into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    for gaze_weight in tqdm(param_grid['gaze_weight'], desc="Optimizing gaze_weight"):
        for prox_epsilon in param_grid['prox_epsilon']:
            # Modify the training data with current parameter values
            train_df['gaze_weight'] = gaze_weight
            train_df['prox_weight'] = 1 - gaze_weight
            train_df['prox_epsilon'] = prox_epsilon

            test_df['gaze_weight'] = gaze_weight
            test_df['prox_weight'] = 1 - gaze_weight
            test_df['prox_epsilon'] = prox_epsilon

            # Define training features and labels
            X_train = train_df[['prox_epsilon', 'gaze_weight']]
            y_train = train_df['grace_eng']

            X_test = test_df[['prox_epsilon', 'gaze_weight']]
            y_test = test_df['grace_eng']

            # Train the model
            model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)

            # Predict and evaluate on the test set
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            # Update best parameters if current MSE is lower
            if mse < best_mse:
                best_mse = mse
                best_params = {'gaze_weight': gaze_weight, 'prox_epsilon': prox_epsilon}

    return best_params, best_mse

def main():
    # Load the dataset
    df = load_data('dataset.csv')  # Replace with your actual dataset file path

    # Parameter grids for optimization
    param_grid = {
        'gaze_weight': np.linspace(0, 1, 10),  # Adjust range and resolution as needed
        'prox_epsilon': np.linspace(0, 1, 10)
    }

    # Optimize parameters
    best_params, best_mse = optimize_parameters(df, param_grid)

    # Print the best parameters and MSE
    print("Optimal parameters found:")
    print(f"gaze_weight: {best_params['gaze_weight']}")
    print(f"prox_epsilon: {best_params['prox_epsilon']}")
    print(f"Test MSE: {best_mse}")

if __name__ == "__main__":
    main()
