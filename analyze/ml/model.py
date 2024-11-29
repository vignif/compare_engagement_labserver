# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np

class AdjusterNN(nn.Module):
    def __init__(self):
        super(AdjusterNN, self).__init__()
        self.fc1 = nn.Linear(3, 50)  # Adjust input size to 3 parameters (prox_epsilon, prox_weight, gaze_weight)
        self.fc2 = nn.Linear(50, 1)  # Output 1 value for grace_eng prediction

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model and move to the appropriate device
def get_model(device='cpu'):
    model = AdjusterNN().to(device)
    return model


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
    


def set_seed(seed=42):
    """
    Sets seed for reproducibility across torch, numpy, and random libraries.
    
    Parameters:
        seed (int): The seed value to set for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior in cuDNN
    torch.backends.cudnn.benchmark = False  # Disables optimizations for reproducibility