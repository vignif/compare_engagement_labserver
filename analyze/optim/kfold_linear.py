import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm  # Import tqdm for the progress bar

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path, sep=';')

def optimize_parameters(df, n_splits=5):
    best_mse = float('inf')
    best_params = None

    # KFold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Wrap the outer loops with tqdm to show progress
    for prox_epsilon in tqdm(df['prox_epsilon'].unique(), desc="Prox Epsilon"):
        for prox_weight in tqdm(df['prox_weight'].unique(), desc="Prox Weight", leave=False):
            for gaze_weight in tqdm(df['gaze_weight'].unique(), desc="Gaze Weight", leave=False):
                fold_mse = []

                for train_idx, val_idx in kf.split(df):
                    train_df = df.iloc[train_idx]
                    val_df = df.iloc[val_idx]

                    # Train a linear regression model
                    X_train = train_df[['prox_epsilon', 'prox_weight', 'gaze_weight']]
                    y_train = train_df['grace_eng']
                    model = LinearRegression()
                    model.fit(X_train, y_train)

                    # Validate on the validation fold
                    X_val = val_df[['prox_epsilon', 'prox_weight', 'gaze_weight']]
                    y_val = val_df['grace_eng']
                    y_pred = model.predict(X_val)

                    # Calculate MSE for this fold
                    fold_mse.append(mean_squared_error(y_val, y_pred))

                # Average MSE across folds
                average_mse = np.mean(fold_mse)

                if average_mse < best_mse:
                    best_mse = average_mse
                    best_params = (prox_epsilon, prox_weight, gaze_weight)

    return best_params, best_mse

def main():
    # Load the dataset
    df = load_data('dataset.csv')  # Replace 'dataset.csv' with your actual dataset file path
    
    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Optimize parameters based on training data
    best_params, best_mse = optimize_parameters(train_df)

    # Evaluate on the testing set using the best parameters
    model = LinearRegression()
    X_test = test_df[['prox_epsilon', 'prox_weight', 'gaze_weight']]
    y_test = test_df['grace_eng']
    model.fit(train_df[['prox_epsilon', 'prox_weight', 'gaze_weight']], train_df['grace_eng'])
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print("Optimal parameters found:")
    print(f"prox_epsilon: {best_params[0]}")
    print(f"prox_weight: {best_params[1]}")
    print(f"gaze_weight: {best_params[2]}")
    print(f"Average MSE from cross-validation: {best_mse}")
    print(f"Test MSE: {test_mse}")

if __name__ == "__main__":
    main()
