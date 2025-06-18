# -*- coding: utf-8 -*-
"""Research_SolarEnergy_02272025_beer_lambert.ipynb


"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/merged_solar_pollution_interpolated.csv'
df = pd.read_csv(file_path)

# Feature Engineering for PGNN
# PGNN에서는 일사량과 태양고도각을 비선형 변환하여 추가적인 물리적 특징을 반영

df["일사량_제곱"] = df["일사"] ** 2
features = ["태양방위각", "태양고도각", "기온", "습도", "일사", "전운량", "1시간평균 미세먼지농도(㎍/㎥)", "일사량_제곱"]
target = "발전량(Target)"

X = df[features].values
y = df[target].values.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)
y_test_actual = scaler_y.inverse_transform(y_test)  # Ensure y_test_actual is defined

# Define Beer-Lambert Physics Loss Function
def beer_lambert_loss(pred, X, alpha=0.01):
    """Physics loss enforcing Beer-Lambert law for light attenuation due to air pollution."""
    I_0 = X[:, 4]  # Measured solar irradiance
    C = X[:, 6]  # PM2.5 concentration
    expected_I = I_0 * torch.exp(-alpha * C)  # Expected irradiance from Beer-Lambert law
    return torch.mean((pred.squeeze() - expected_I) ** 2)

# Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define PGNN Model
class PGNN(nn.Module):
    def __init__(self, input_dim):
        super(PGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loss(self, y_pred, y_true, X):
        mse = nn.MSELoss()(y_pred, y_true)
        phy_loss = beer_lambert_loss(y_pred, X)
        return 0.8 * mse + 0.2 * phy_loss  # PGNN with Beer-Lambert physics loss

# Define PINN Model
class PINN(PGNN):
    def __init__(self, input_dim):
        super(PINN, self).__init__(input_dim)

    def loss(self, y_pred, y_true, X):
        mse = nn.MSELoss()(y_pred, y_true)
        phy_loss = beer_lambert_loss(y_pred, X)
        return 0.5 * mse + 0.5 * phy_loss  # PINN with stronger physics constraint

# Define LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence length dimension
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Training function
def train_model(model, model_name, num_epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_torch)
        if isinstance(model, (PINN, PGNN)):
            loss = model.loss(y_pred, y_train_torch, X_train_torch)
        else:
            loss = mse_loss(y_pred, y_train_torch)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"{model_name} Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"{model_name} model saved.")

# Train all models
pgnn_model = PGNN(input_dim=X_train.shape[1])
pinn_model = PINN(input_dim=X_train.shape[1])
mlp_model = MLP(input_dim=X_train.shape[1])
lstm_model = LSTM(input_dim=X_train.shape[1])

train_model(pgnn_model, "PGNN")
train_model(pinn_model, "PINN")
train_model(mlp_model, "MLP")
train_model(lstm_model, "LSTM")

# Load and evaluate models
def evaluate_model(model, model_name):
    model.load_state_dict(torch.load(f"{model_name}.pth"))
    model.eval()
    y_pred_test = model(X_test_torch).detach().numpy()
    y_pred_test = scaler_y.inverse_transform(y_pred_test)
    mse = np.mean((y_pred_test - y_test_actual) ** 2)
    print(f"{model_name} Test MSE: {mse:.4f}")
    return y_pred_test

pgnn_preds = evaluate_model(pgnn_model, "PGNN")
pinn_preds = evaluate_model(pinn_model, "PINN")
mlp_preds = evaluate_model(mlp_model, "MLP")
lstm_preds = evaluate_model(lstm_model, "LSTM")

# Enhanced Visualization
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:100], label="Actual", color='black', linewidth=2)
plt.plot(pgnn_preds[:100], label="PGNN", linestyle='dashed', marker='o', markersize=4)
plt.plot(pinn_preds[:100], label="PINN", linestyle='dotted', marker='s', markersize=4)
plt.plot(mlp_preds[:100], label="MLP", linestyle='dashdot', marker='^', markersize=4)
plt.plot(lstm_preds[:100], label="LSTM", linestyle='solid', marker='x', markersize=4)
plt.fill_between(range(100), pgnn_preds[:100].flatten(), pinn_preds[:100].flatten(), color='gray', alpha=0.2, label="PGNN-PINN Range")
plt.legend()
plt.title("Model Predictions vs Actual")
plt.xlabel("Samples")
plt.ylabel("Power Output")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Generate Performance Table
mse_pgnn = np.mean((pgnn_preds - y_test_actual) ** 2)
mse_pinn = np.mean((pinn_preds - y_test_actual) ** 2)
mse_mlp = np.mean((mlp_preds - y_test_actual) ** 2)
mse_lstm = np.mean((lstm_preds - y_test_actual) ** 2)

performance_df = pd.DataFrame({
    "Model": ["PGNN", "PINN", "MLP", "LSTM"],
    "MSE": [mse_pgnn, mse_pinn, mse_mlp, mse_lstm]
})

print("\nModel Performance Comparison:")
print(performance_df)

