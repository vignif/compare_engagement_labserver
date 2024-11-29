import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def optimize_parameters(df, n_splits=5):
    """Optimize parameters for the Random Forest model using K-Fold cross-validation."""
    best_mse = float('inf')
    best_params = None
    learning_steps = []  # List to store learning steps
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    for prox_epsilon in tqdm(df['prox_epsilon'].unique(), desc="Prox Epsilon"):
        for prox_weight in tqdm(df['prox_weight'].unique(), desc="Prox Weight", leave=False):
            for gaze_weight in tqdm(df['gaze_weight'].unique(), desc="Gaze Weight", leave=False):
                fold_mse = []

                for train_idx, val_idx in kf.split(df):
                    train_df = df.iloc[train_idx]
                    val_df = df.iloc[val_idx]

                    # Prepare training data
                    X_train = train_df[['prox_epsilon', 'prox_weight', 'gaze_weight']]
                    y_train = train_df['eng_1']  # Change target variable to eng_1

                    # Hyperparameter tuning using GridSearchCV
                    model = RandomForestRegressor(random_state=42)
                    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
                    grid_search.fit(X_train, y_train)

                    # Get the best model from grid search
                    best_model = grid_search.best_estimator_

                    # Validate on the validation fold
                    X_val = val_df[['prox_epsilon', 'prox_weight', 'gaze_weight']]
                    y_val = val_df['eng_1']  # Change target variable to eng_1
                    y_pred = best_model.predict(X_val)

                    # Calculate MSE for this fold
                    fold_mse.append(mean_squared_error(y_val, y_pred))

                # Average MSE across folds
                average_mse = np.mean(fold_mse)
                learning_steps.append((prox_epsilon, prox_weight, gaze_weight, average_mse))  # Store parameters and MSE

                if average_mse < best_mse:
                    best_mse = average_mse
                    best_params = (prox_epsilon, prox_weight, gaze_weight)

    return best_params, best_mse, learning_steps  # Return learning steps
