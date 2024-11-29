import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.flatten()

# Load the best model
model = joblib.load("xgb_best_model.joblib")

# Perform predictions
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.4f}")

# Optionally plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # 45-degree line
plt.xlabel("Actual grace_eng")
plt.ylabel("Predicted grace_eng")
plt.title("Actual vs Predicted grace_eng")
plt.grid(True)
plt.savefig("predicted_vs_actual.png")
plt.close()

print("Model evaluation complete.")
