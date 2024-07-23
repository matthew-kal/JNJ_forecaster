import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from copy import deepcopy as dc
import joblib

# Load data
ticker = "JNJ"
df = yf.download(ticker, start='2000-01-01', end='2024-06-30')[['Close']]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Prepare data
def prep_df(df, steps):
    df = df.copy()
    for i in range(1, steps + 1):
        df[f'C(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

lb = 5  # lookback period
data = prep_df(df, lb).to_numpy()

scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data)

X, y = data[:, 1:], data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42, shuffle=True)

X_train = X_train.reshape((-1, lb, 1))
X_test = X_test.reshape((-1, lb, 1))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# Define Dataset class
class TSD(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_ds = TSD(X_train, y_train)
test_ds = TSD(X_test, y_test)

bs = 16  # batch size
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False)

# Define LSTM model with additional layers and hyperparameter tuning
class LSTM(nn.Module):
    def __init__(self, hs, drop=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(1, hs, 1, batch_first=True, dropout=drop)
        self.lstm2 = nn.LSTM(hs, hs, 1, batch_first=True, dropout=drop)
        self.fc1 = nn.Linear(hs, hs // 2)
        self.fc2 = nn.Linear(hs // 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hs).to(device)
        c0 = torch.zeros(1, x.size(0), hs).to(device)
        out, _ = self.lstm1(x, (h0, c0))
        out, _ = self.lstm2(out, (h0, c0))
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out

hs = 100  # hidden size
model = LSTM(hs)
model.to(device)

# Training function
def train_epoch():
    model.train()
    run_loss = 0.0
    for i, (x, y) in enumerate(train_dl):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        run_loss += loss.item()
        loss.backward()
        opt.step()
        if i % 100 == 99:
            avg_loss = run_loss / 100
            print(f'Batch {i + 1}, Loss: {avg_loss:.3f}')
            run_loss = 0.0

# Validation function
def val_epoch():
    model.eval()
    run_loss = 0.0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            run_loss += loss.item()
    avg_loss = run_loss / len(test_dl)
    print(f'Val Loss: {avg_loss:.3f}')
    print('***************************************************')

lr = 0.001  # learning rate
epochs = 25  # increased epochs for better training
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    print(f'Epoch: {epoch + 1}')
    train_epoch()
    val_epoch()

# Save the model and scaler
torch.save(model.state_dict(), 'lstm_model.pth')
joblib.dump(scaler, 'scaler.pkl')

# Evaluate model performance
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy()
    y_true = y_test.cpu().numpy()

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f'Mean Absolute Error: {mae:.4f}')
print(f'Mean Squared Error: {mse:.4f}')
print(f'R^2 Score: {r2:.4f}')
