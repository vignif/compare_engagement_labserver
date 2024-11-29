import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import get_model, CustomDataset, set_seed

# Set seed for reproducibility
set_seed(42)

# Training function
def train_model(num_epochs=3, learning_rate=0.001, batch_size=64):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device).to(device)

    # Load and split the dataset
    dataset = CustomDataset('dataset.csv')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0  # Track the loss for the epoch
        for x, y_grace, y_target in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            x, y_grace, y_target = x.to(device), y_grace.to(device), y_target.to(device)

            # Forward pass
            predicted_grace_eng = model(x).squeeze()
            loss = criterion(predicted_grace_eng, y_target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate the loss

        avg_epoch_loss = epoch_loss / len(train_loader)  # Average loss for the epoch
        print(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as model.pth")
    return model, val_loader  # Return the model and validation loader for evaluation

# Run the training function if the script is executed directly
if __name__ == "__main__":
    train_model(num_epochs=10)
