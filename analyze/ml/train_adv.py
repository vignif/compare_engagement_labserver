import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import get_model, CustomDataset, set_seed

# Set seed for reproducibility
set_seed(42)

# Training function
def train_model(num_epochs=10, learning_rate=0.001, batch_size=64):
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
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Early stopping variables
    best_val_loss = float('inf')
    patience = 5
    counter = 0

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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate the loss

        avg_epoch_loss = epoch_loss / len(train_loader)  # Average loss for the epoch
        print(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss:.4f}")

        # Validate model and adjust learning rate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, _, y_target_val in val_loader:
                x_val, y_target_val = x_val.to(device), y_target_val.to(device)
                predicted_val = model(x_val).squeeze()
                val_loss += criterion(predicted_val, y_target_val).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Step the learning rate scheduler
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved as best_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    return model  # Return the trained model

# Run the training function if the script is executed directly
if __name__ == "__main__":
    train_model()
