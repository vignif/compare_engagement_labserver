import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import cupy as cp
import time
import matplotlib.pyplot as plt
import optuna

# Load dataset
data = pd.read_csv('dataset.csv', sep=';')  # Update with your actual file path

# Feature and target preparation
X = data[['prox_epsilon', 'prox_weight', 'gaze_weight']]
y = data['grace_eng']

# Check for CUDA availability
if cp.cuda.runtime.getDeviceCount() > 0:
    print("CUDA is available. Proceeding with GPU acceleration.")
    use_gpu = True
else:
    print("CUDA is not available. The model will run on CPU.")
    use_gpu = False

# Function to optimize
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.3)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    
    # Initialize the model with suggested parameters
    model = xgb.XGBRegressor(
        tree_method='gpu_hist' if use_gpu else 'hist',  # Use GPU if available
        objective='reg:squarederror',
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        eval_metric='rmse'
    )

    # Perform cross-validation
    kf = KFold(n_splits=5)
    rmse_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# Optimize hyperparameters with Optuna
start_time = time.time()
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed
end_time = time.time()

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)
print("Best RMSE:", study.best_value)
print("Optimization took {:.2f} seconds.".format(end_time - start_time))

# Train final model with best parameters
best_params = study.best_params
final_model = xgb.XGBRegressor(
    tree_method='gpu_hist' if use_gpu else 'hist',
    objective='reg:squarederror',
    **best_params
)

# Split the dataset for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
final_model.fit(X_train, y_train)
y_test_pred = final_model.predict(X_test)

# Evaluate the final model
final_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
print("Final RMSE on test set:", final_rmse)

# Plotting true vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs Predicted Values")
plt.savefig("true_vs_predicted.png")  # Save the figure
plt.show()
