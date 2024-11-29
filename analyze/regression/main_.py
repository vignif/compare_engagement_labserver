import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
import time

# Load the dataset
df = pd.read_csv('dataset.csv', sep=';')

# Define features (X) and target (y)
X = df[['prox_epsilon', 'prox_weight', 'gaze_weight']]
y = df['eng_1']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Set up an extended parameter grid for hyperparameter tuning
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize Random Forest model
model = RandomForestRegressor(random_state=42)

# Initialize RandomizedSearchCV for efficient hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=model, param_distributions=param_dist, n_iter=30, 
    scoring='neg_mean_squared_error', cv=kf, verbose=1, random_state=42, n_jobs=-1
)

# Track the time taken by RandomizedSearchCV
print("Starting hyperparameter tuning with RandomizedSearchCV...")
start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()
print(f"Hyperparameter tuning completed in {end_time - start_time:.2f} seconds.")

# Get the best model from RandomizedSearchCV
best_model = random_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate the Mean Squared Error and print optimal parameters
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on the Test Set: {mse}")
print("Optimal parameters found:")
print(random_search.best_params_)

# Plotting Predicted vs Actual values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Predicted')
sns.lineplot(x=y_test, y=y_test, color='red', label='Ideal Line')
plt.xlabel('Actual eng_1')
plt.ylabel('Predicted eng_1')
plt.title('Actual vs Predicted eng_1 Values')
plt.legend()
plt.savefig('actual_predicted.png')

# Plotting Feature Importances
importances = best_model.feature_importances_
features = ['prox_epsilon', 'prox_weight', 'gaze_weight']
plt.figure(figsize=(8, 6))
sns.barplot(x=features, y=importances)
plt.title('Feature Importance')
plt.savefig('importance.png')

# Save the scaler and model for future use
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(best_model, 'optimized_model.pkl')

# Print out the total execution time for each process in a structured way
print(f"Data Preprocessing Time: {time.time() - start_time:.2f} seconds")
print(f"Model Training Time: {end_time - start_time:.2f} seconds")
