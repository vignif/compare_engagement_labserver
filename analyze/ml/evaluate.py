import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from model import get_model, CustomDataset
import numpy as np

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Load dataset
    dataset = CustomDataset('dataset.csv')
    
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for validation set
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    criterion = torch.nn.MSELoss()
    val_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for x, y_grace, y_target in val_loader:
            x, y_grace, y_target = x.to(device), y_grace.to(device), y_target.to(device)

            # Forward pass
            predicted_grace_eng = model(x).squeeze()
            val_loss += criterion(predicted_grace_eng, y_target).item()

            # Collect predictions and targets for plotting
            predictions.append(predicted_grace_eng.cpu().numpy())
            targets.append(y_target.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Convert lists to numpy arrays
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Scatter plot of predictions vs actual
    plt.subplot(1, 2, 1)
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')  # diagonal line
    plt.title('Predictions vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.axis('equal')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.title('Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()
