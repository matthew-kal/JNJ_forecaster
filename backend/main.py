from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hs = 100  
model = LSTM(hs)
model.load_state_dict(torch.load('lstm_model.pth', map_location=device))
model.to(device)
model.eval()

scaler = joblib.load('scaler.pkl')

class StockRequest(BaseModel):
    ticker: str

def prep_df(df, steps):
    df = df.copy()
    for i in range(1, steps + 1):
        df[f'C(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

@app.post("/predict")
def predict(request: StockRequest):
    ticker = request.ticker
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=10)  
    df = yf.download(ticker, start=start_date, end=end_date)[['Close']]
    initial_length = len(df)

    if initial_length < 6:
        start_date = end_date - timedelta(days=20)  
        df = yf.download(ticker, start=start_date, end=end_date)[['Close']]
    

    if df.empty:
        return {"error": "No data received from Yahoo Finance"}

    lb = 5  
    df_prepared = prep_df(df, lb)

    if df_prepared.empty:
        return {"error": "Not enough data after processing to make a prediction"}

    data = df_prepared.to_numpy()
    data = scaler.transform(data)

    X = data[:, 1:]
    X = X.reshape((-1, lb, 1))
    X = torch.tensor(X).float().to(device)

    results = df['Close'].values[-lb:].tolist()

    with torch.no_grad():
        pred = model(X[-1].unsqueeze(0)).cpu().numpy().flatten()
        results.append(pred[0])

    dummy = np.zeros((1, data.shape[1]))
    dummy[0, 0] = results[-1]
    dummy = scaler.inverse_transform(dummy)
    results[-1] = dummy[0, 0]

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



