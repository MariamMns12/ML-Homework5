# Homework 5 - Problem 2
# Linear Regression on Housing Dataset
# Author: Mariam Mahmoud (mariammns12)
# ECGR 4105 - Machine Learning for Engineers

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ----------------------------
# Load the housing dataset
# ----------------------------
# Make sure your dataset file is in the same folder as this script
# Example dataset columns: area, bedrooms, bathrooms, stories, parking, price
data = pd.read_csv("Housing.csv")

# Select features and target
X = data[["area", "bedrooms", "bathrooms", "stories", "parking"]].values
y = data["price"].values

# Normalize features for stability
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Define Linear Model
# ----------------------------
model = torch.nn.Linear(5, 1)  # 5 inputs → 1 output

# ----------------------------
# Training Function
# ----------------------------
def train_model(learning_rate, n_epochs=5000):
    torch.manual_seed(42)
    model = torch.nn.Linear(5, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        # Forward pass
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = loss_fn(y_val_pred, y_val)
        
        # Store every 500 epochs
        if epoch % 500 == 0:
            print(f"LR={learning_rate} | Epoch={epoch:5d} | Train Loss={loss.item():10.4f} | Val Loss={val_loss.item():10.4f}")
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())

    return model, train_losses, val_losses

# ----------------------------
# Train with Different Learning Rates
# ----------------------------
learning_rates = [0.1, 0.01, 0.001, 0.0001]
results = {}

for lr in learning_rates:
    print(f"\n--- Training Linear Regression with LR={lr} ---")
    model, train_losses, val_losses = train_model(lr)
    results[lr] = {
        "train_loss": train_losses[-1] if train_losses else None,
        "val_loss": val_losses[-1] if val_losses else None
    }

# ----------------------------
# Display Results
# ----------------------------
print("\nFinal Results Summary:")
for lr, vals in results.items():
    print(f"LR={lr:<7} | Train Loss={vals['train_loss']:<10.4f} | Val Loss={vals['val_loss']:<10.4f}")

# Plot last model's training vs validation loss (optional)
plt.figure(figsize=(6,4))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Training vs Validation Loss (last model)")
plt.xlabel("Checkpoint (x500 epochs)")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig("housing_linear_loss_plot.png")
plt.show()
