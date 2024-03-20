from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from pathlib import Path
import numpy as np
from helper_functions import plot_decision_boundary

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Generate dataset
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# # Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # Input layer
        self.fc2 = nn.Linear(10, 10)  # Hidden layer
        self.fc3 = nn.Linear(10, 1)   # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = MyModel()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

def train_model(model, criterion, optimizer, X_train, y_train, epochs=100):
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                train_acc = torch.sum((outputs > 0.5) == y_train.view(-1, 1)).item() / len(y_train)
                test_outputs = model(X_test)
                test_acc = torch.sum((test_outputs > 0.5) == y_test.view(-1, 1)).item() / len(y_test)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

train_model(model, criterion, optimizer, X_train, y_train)


# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)


# print(X[:10], y[:10])