import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("dataset.csv", sep=";")

# Features and target
X = data[['prox_epsilon', 'prox_weight', 'gaze_weight']]
y = data['eng_1']

# Cross-validation setup
n_splits = 5

# Store performance results
fold_results = []

print("Evaluating saved XGBoost models...")
for fold in range(n_splits):
    # Load the model for the current fold
    model_path = f"xgb_model_fold_{fold}.joblib"
    model = joblib.load(model_path)
    
    # Perform predictions using the entire dataset for evaluation
    y_pred = model.predict(X)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    fold_results.append(rmse)
    
    print(f"Fold {fold + 1} RMSE: {rmse:.4f}")
    
    # Plotting the predictions vs actual for this fold
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.3)
    plt.xlabel("Actual eng_1")
    plt.ylabel("Predicted eng_1")
    plt.title(f"Predictions vs Actuals - Fold {fold + 1}")
    plt.grid(True)
    plt.savefig(f"xgb_predictions_vs_actuals_fold_{fold + 1}.png")
    plt.close()  # Close the plot to avoid display during loops

# Report overall performance
mean_rmse = np.mean(fold_results)
print(f"\nOverall Mean RMSE across all folds: {mean_rmse:.4f}")
