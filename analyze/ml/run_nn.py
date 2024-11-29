import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from torch.utils.data import random_split


# Custom Dataset to load your data
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file, delimiter=';')
        self.features = data[['prox_epsilon', 'prox_weight', 'gaze_weight']].values
        self.grace_eng = data['grace_eng'].values
        self.eng_1 = data['eng_1'].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y_grace = torch.tensor(self.grace_eng[idx], dtype=torch.float32)
        y_target = torch.tensor(self.eng_1[idx], dtype=torch.float32)
        return x, y_grace, y_target

# Define the neural network
class AdjusterNN(nn.Module):
    def __init__(self):
        super(AdjusterNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Load dataset and create DataLoader
dataset = CustomDataset('dataset.csv')  # Replace 'data.csv' with the actual path
loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

# Model, loss, and optimizer
model = AdjusterNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(3):  # Adjust number of epochs as needed
    for x, y_grace, y_target in tqdm(loader):
        x, y_grace, y_target = x.to(device), y_grace.to(device), y_target.to(device)

        # Forward pass
        predicted_grace_eng = model(x).squeeze()
        loss = criterion(predicted_grace_eng, y_target)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Check how well grace_eng aligns with eng_1 after training
with torch.no_grad():
    total_diff = 0
    for x, y_grace, y_target in loader:
        x, y_grace, y_target = x.to(device), y_grace.to(device), y_target.to(device)
        predicted_grace_eng = model(x).squeeze()
        total_diff += torch.sum(torch.abs(predicted_grace_eng - y_target))

    print(f"Average alignment error: {total_diff / len(dataset):.4f}")
