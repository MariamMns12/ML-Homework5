# Temperature Prediction - Nonlinear vs Linear Model (Fixed Version)
# Author: Mariam Mahmoud (mariammns12)
# ECGR 4105 - Homework 5

import torch
import matplotlib.pyplot as plt

# ---------------------------
# Dataset (Unnormalized Input and Target)
# ---------------------------
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])

# ---------------------------
# Normalize the input to prevent exploding gradients
# ---------------------------
t_un = 0.1 * t_u  # scaled input for numerical stability

# ---------------------------
# Nonlinear Model Definition
# ---------------------------
def model(t_u, w1, w2, b):
    return w2 * (t_u ** 2) + w1 * t_u + b

# ---------------------------
# Linear Model Definition (Baseline)
# ---------------------------
def linear_model(t_u, w, b):
    return w * t_u + b

# ---------------------------
# Training Loop
# ---------------------------
def training_loop(n_epochs, learning_rate):
    w1 = torch.ones(1, requires_grad=True)
    w2 = torch.ones(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.SGD([w1, w2, b], lr=learning_rate)
    for epoch in range(n_epochs):
        t_p = model(t_un, w1, w2, b)  # use normalized input
        loss = torch.nn.functional.mse_loss(t_p, t_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():10.4f}")

    return w1, w2, b

# ---------------------------
# Training Nonlinear Model for Different Learning Rates
# ---------------------------
learning_rates = [0.1, 0.01, 0.001, 0.0001]
results = {}

for lr in learning_rates:
    print(f"\nTraining with learning rate = {lr}")
    w1, w2, b = training_loop(5000, lr)
    final_loss = torch.nn.functional.mse_loss(model(t_un, w1, w2, b), t_c)
    results[lr] = (w1.item(), w2.item(), b.item(), final_loss.item())

print("\nFinal Nonlinear Model Results:")
for lr, (w1, w2, b, loss) in results.items():
    print(f"LR={lr:<7} w1={w1:8.4f} w2={w2:10.6f} b={b:8.4f}  Loss={loss:10.4f}")

# ---------------------------
# Pick the best (lowest loss) model
# ---------------------------
best_lr = min(results, key=lambda lr: results[lr][3])
best_w1, best_w2, best_b, best_loss = results[best_lr]
print(f"\n✅ Best Model -> LR={best_lr} | Loss={best_loss:.4f}")
print(f"   Parameters -> w1={best_w1:.4f}, w2={best_w2:.6f}, b={best_b:.4f}")

# ---------------------------
# Linear Baseline (from Lecture Example)
# ---------------------------
baseline_w = 0.45
baseline_b = -17.0

# ---------------------------
# Predictions and Plot
# ---------------------------
t_p_linear = linear_model(t_un, baseline_w, baseline_b)
t_p_nonlinear = model(t_un, best_w1, best_w2, best_b)

plt.figure(figsize=(8, 5))
plt.scatter(t_un * 10, t_c, label="Actual Data", color="blue")  # rescale input for plot
plt.plot(t_un * 10, t_p_linear, label="Linear Model", color="orange")
plt.plot(t_un * 10, t_p_nonlinear, label="Nonlinear Model", color="green")
plt.xlabel("Unnormalized Temperature (t_u)")
plt.ylabel("Celsius Temperature (t_c)")
plt.title("Linear vs Nonlinear Model Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot for your report
plt.savefig("comparison_plot.png")
plt.show()
