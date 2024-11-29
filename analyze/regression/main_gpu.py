import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("dataset.csv", sep=";")

# Features and target
X = data[['prox_epsilon', 'prox_weight', 'gaze_weight']]
y = data['eng_1']

# Cross-validation setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Track training time and performance
cv_results = []
params_per_fold = []

print("Starting cross-validation with XGBoost...")
with tqdm(total=n_splits, desc="Cross-validation") as pbar:
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Set up XGBoost with GPU support
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',  # Use hist for optimized histogram-based training
            device='cuda',       # Specify CUDA device for GPU acceleration
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6
        )
        
        print(f"Training fold {fold + 1}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        cv_results.append(rmse)
        
        # Save model for each fold
        model_path = f"xgb_model_fold_{fold}.joblib"
        joblib.dump(model, model_path)
        print(f"Fold {fold + 1} model saved as '{model_path}' with RMSE: {rmse:.4f}")
        
        # Store parameters used in this fold
        params_per_fold.append(model.get_params())
        
        # Update progress bar
        pbar.update(1)

# Report cross-validation performance
mean_rmse = np.mean(cv_results)
print(f"\nCross-validation complete. Mean RMSE across folds: {mean_rmse:.4f}")
print("\nParameters used for each fold:")
for fold, params in enumerate(params_per_fold):
    print(f"Fold {fold + 1}: {params}")

# Save results plot
print("Generating prediction vs actual plot...")
y_pred_plot = model.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_plot, alpha=0.3)
plt.xlabel("Actual eng_1")
plt.ylabel("Predicted eng_1")
plt.title("XGBoost Predictions vs Actuals")
plt.savefig("xgb_predictions_vs_actuals.png")
print("Plot saved as 'xgb_predictions_vs_actuals.png'")
plt.show()
