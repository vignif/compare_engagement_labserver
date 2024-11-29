import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("dataset.csv", sep=";")

# Features and target
X = data[['prox_epsilon', 'prox_weight', 'gaze_weight']]
y = data['grace_eng']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the training and test sets for later use
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Data preprocessing complete. Training and test sets saved.")
