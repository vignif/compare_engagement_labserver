import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib

# Load training data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.flatten()

# Set up cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store the best parameters
best_params = None
best_rmse = float('inf')

for learning_rate in [0.01, 0.1, 0.2]:
    for max_depth in [3, 5, 7]:
        fold_rmse = []
        
        for train_index, val_index in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_tr, y_val = y_train[train_index], y_train[val_index]

            # Train the model
            model = xgb.XGBRegressor(
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=1000,
                objective='reg:squarederror'
            )
            model.fit(X_tr, y_tr)

            # Validate the model
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            fold_rmse.append(rmse)

        mean_rmse = np.mean(fold_rmse)
        print(f"Learning Rate: {learning_rate}, Max Depth: {max_depth}, RMSE: {mean_rmse:.4f}")
        
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = (learning_rate, max_depth)

# Save the best model with the best parameters
best_model = xgb.XGBRegressor(
    learning_rate=best_params[0],
    max_depth=best_params[1],
    n_estimators=1000,
    objective='reg:squarederror'
)
best_model.fit(X_train, y_train)
joblib.dump(best_model, "xgb_best_model.joblib")

print(f"Best parameters: Learning Rate: {best_params[0]}, Max Depth: {best_params[1]}")
