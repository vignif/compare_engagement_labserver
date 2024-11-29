import os
import numpy as np
import rosbag
import cv2
from cv_bridge import CvBridge
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import pympi
from get_data import UserData

# ROS and image preprocessing
bridge = CvBridge()


# Define the LSTM model
class EngagementModel(nn.Module):
    def __init__(self):
        super(EngagementModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the last classification layer
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze ResNet layers

        self.lstm = nn.LSTM(input_size=2048, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.resnet(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])
        return torch.sigmoid(out).squeeze()

# Training and data processing
def prepare_data(file_list):
    """Load data for each name in file_list."""
    user_data = []
    for name in file_list:
        data = UserData(name.strip())
        if data.images:
            user_data.append(data)
    return user_data

def train_model(user_data, epochs=10):
    """Train the LSTM model on processed images and engagement data."""
    model = EngagementModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for data in user_data:
            images = np.array([img['image'] for img in data.images])
            labels = torch.tensor(data.engagement, dtype=torch.float32)

            images = torch.tensor(images).permute(0, 3, 1, 2).float() / 255.0
            images = images.unsqueeze(0)  # Add batch dimension
            labels = labels.unsqueeze(0)  # Add batch dimension

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    print("Training complete!")

def main():
    # Read list of bag names
    with open('mono_welcome', 'r') as fp:
        file_list = fp.readlines()

    # Prepare data and train
    user_data = prepare_data(file_list)
    train_model(user_data)

if __name__ == '__main__':
    main()
