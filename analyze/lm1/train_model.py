import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_and_evaluate(train_df, val_df, test_df, best_params):
    """Train the Random Forest model and evaluate its performance."""
    model = RandomForestRegressor(random_state=42)
    model.set_params(**best_params)  # Use the best parameters found
    model.fit(train_df[['prox_epsilon', 'prox_weight', 'gaze_weight']], train_df['eng_1'])  # Train on eng_1

    # Evaluate on the validation set
    X_val = val_df[['prox_epsilon', 'prox_weight', 'gaze_weight']]
    y_val = val_df['eng_1']
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)

    # Evaluate on the testing set
    X_test = test_df[['prox_epsilon', 'prox_weight', 'gaze_weight']]
    y_test = test_df['eng_1']
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)

    return val_mse, test_mse
