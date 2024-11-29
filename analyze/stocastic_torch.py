import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Load your engagement data from the CSV file
csv_file = "ds.csv"  # Replace with your file name or path
prox_weight_column = "prox_weight"  # Replace with the name of the prox_weight column
grace_prox_column = "grace_prox"   # Replace with the name of the grace_prox column
gaze_weight_column = "gaze_weight" # Replace with the name of the gaze_weight column
grace_gaze_column = "grace_gaze"   # Replace with the name of the grace_gaze column
eng_4_column = "eng_1"             # Replace with the name of the target column (eng_4)

# Load the dataset
data = pd.read_csv(csv_file, sep=';')

# Prepare the input features and target
prox_weight = torch.tensor(data[prox_weight_column].values, dtype=torch.float32).view(-1, 1)
grace_prox = torch.tensor(data[grace_prox_column].values, dtype=torch.float32).view(-1, 1)
gaze_weight = torch.tensor(data[gaze_weight_column].values, dtype=torch.float32).view(-1, 1)
grace_gaze = torch.tensor(data[grace_gaze_column].values, dtype=torch.float32).view(-1, 1)
eng_4 = torch.tensor(data[eng_4_column].values, dtype=torch.float32).view(-1, 1)  # Target column

# Normalize the inputs to have similar scale using PyTorch
prox_weight = (prox_weight - prox_weight.mean()) / prox_weight.std()
grace_prox = (grace_prox - grace_prox.mean()) / grace_prox.std()
gaze_weight = (gaze_weight - gaze_weight.mean()) / gaze_weight.std()
grace_gaze = (grace_gaze - grace_gaze.mean()) / grace_gaze.std()

# Create the dataset and DataLoader for mini-batch processing
dataset = TensorDataset(prox_weight, grace_prox, gaze_weight, grace_gaze, eng_4)
batch_size = 64  # Larger batch size for better efficiency
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model function: grace is the weighted sum of prox_weight and gaze_weight
def forward(prox_weight, gaze_weight, grace_prox, grace_gaze, b):
    return prox_weight * grace_prox + gaze_weight * grace_gaze + b

# Loss function (Mean Squared Error)
def criterion(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

# Regularization term (to encourage weights to sum to 1)
def regularization_loss(w_prox, w_gaze):
    return torch.abs((w_prox + w_gaze) - 1)

# Initialize weights with smaller magnitude
w_prox = torch.tensor(0.5, requires_grad=True)  # Initial prox_weight
w_gaze = torch.tensor(0.5, requires_grad=True)  # Initial gaze_weight
b_BGD = torch.tensor(0.0, requires_grad=True)  # Initial bias

# Set learning rate for Adam optimizer
learning_rate = 0.001

# Create the optimizer (Adam instead of SGD)
optimizer = torch.optim.Adam([w_prox, w_gaze, b_BGD], lr=learning_rate)

# List to store loss values during training
losses_SGD = []

# Number of epochs
n_iter = 100

# Training loop using Adam
for epoch in range(n_iter):
    epoch_loss = 0.0
    # Iterate over mini-batches
    for prox_w, grace_p, gaze_w, grace_g, target in dataloader:
        # Forward pass for this mini-batch
        Y_pred = forward(w_prox, w_gaze, grace_p, grace_g, b_BGD)
        
        # Calculate loss
        loss = criterion(Y_pred, target)  # We want to match the 'eng_4' column
        
        # Add regularization loss to encourage weights to sum to 1
        reg_loss = regularization_loss(w_prox, w_gaze)
        total_loss = loss + 0.1 * reg_loss  # Weight regularization loss by a factor of 0.1

        epoch_loss += total_loss.item()

        # Backward pass for this mini-batch
        optimizer.zero_grad()  # Zero out previous gradients
        total_loss.backward()  # Compute gradients
        
        # Gradient clipping to prevent too large updates
        torch.nn.utils.clip_grad_norm_([w_prox, w_gaze, b_BGD], max_norm=1.0)
        
        # Update the weights
        optimizer.step()

        # Ensure the weights are positive using ReLU
        with torch.no_grad():
            w_prox.data = torch.relu(w_prox.data)  # Apply ReLU to ensure w_prox is positive
            w_gaze.data = torch.relu(w_gaze.data)  # Apply ReLU to ensure w_gaze is positive

            # Normalize the weights so that their sum equals 1
            weight_sum = w_prox + w_gaze
            w_prox.data /= weight_sum
            w_gaze.data /= weight_sum

    # Print the loss every 10 epochs for tracking progress
    if epoch % 10 == 0:
        print(f"Adam Epoch {epoch}, Loss: {epoch_loss / len(dataloader):.4f}, prox_weight: {w_prox.item():.4f}, gaze_weight: {w_gaze.item():.4f}, b: {b_BGD.item():.4f}")

    losses_SGD.append(epoch_loss / len(dataloader))

# Plotting the loss curve
plt.plot(losses_SGD, label="Adam Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Training Loss Over Epochs')
plt.show()
