# Homework 5 - Problem 3
# Neural Network for Housing Price Prediction
# Author: Mariam Mahmoud (mariammns12)
# ECGR 4105 - Machine Learning for Engineers

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

# ------------------------------------
# Load and preprocess dataset
# ------------------------------------
data = pd.read_csv("Housing.csv")
X = data[["area", "bedrooms", "bathrooms", "stories", "parking"]].values
y = data["price"].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------
# Define Neural Network Architectures
# ------------------------------------
class ModelA(nn.Module):  # One hidden layer
    def __init__(self):
        super(ModelA, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.net(x)

class ModelB(nn.Module):  # Three hidden layers
    def __init__(self):
        super(ModelB, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------------
# Training function
# ------------------------------------
def train_model(model, learning_rate=0.001, epochs=200):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, val_losses = [], []
    start = time.time()

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Train Loss={loss.item():.4f} | Val Loss={val_loss.item():.4f}")

    duration = time.time() - start
    return train_losses, val_losses, duration

# ------------------------------------
# Train and evaluate both models
# ------------------------------------
results = {}

for name, Net in {"ModelA": ModelA, "ModelB": ModelB}.items():
    print(f"\n--- Training {name} ---")
    model = Net()
    train_losses, val_losses, duration = train_model(model)
    results[name] = {
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "time": duration
    }

    plt.plot(train_losses, label=f"{name} - Train")
    plt.plot(val_losses, label=f"{name} - Val")

plt.title("Training vs Validation Loss for Neural Networks")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig("housing_nn_loss_plot.png")
plt.show()

# ------------------------------------
# Display Results Summary
# ------------------------------------
print("\nFinal Neural Network Results:")
print("{:<10} {:<15} {:<15} {:<10}".format("Model", "Train Loss", "Val Loss", "Time (s)"))
for name, vals in results.items():
    print(f"{name:<10} {vals['train_loss']:<15.4f} {vals['val_loss']:<15.4f} {vals['time']:<10.2f}")
