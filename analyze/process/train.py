import xgboost as xgb
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

# Load training data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.flatten()

# Create and train the model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "xgb_model.joblib")

print("Model training complete and saved.")
