import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kenya Inflation Forecast", layout="wide")

# ----------------------------
# Model definition (must match training exactly)
# ----------------------------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

SEQ_LEN = 12
FEATURE_COLS = [
    'headline_inflation', 'brent_crude', 'cbk_rate',
    'kes_usd', 'm3', 'forex_reserves', 'month', 'm3_pct_change'
]
TARGET_COL_IDX = FEATURE_COLS.index('headline_inflation')

# ----------------------------
# Cached loaders
# ----------------------------
@st.cache_resource
def load_model():
    model = GRUModel(input_size=len(FEATURE_COLS))
    model.load_state_dict(torch.load('gru_model.pth', map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    with open('scaler_nolags.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_bounds():
    with open('inflation_bounds.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv('Cleaned Dataset.csv', parse_dates=['date'])
    df['month'] = df['date'].dt.month
    df['m3_pct_change'] = df['m3'].pct_change() * 100
    df = df.dropna().reset_index(drop=True)
    return df

# ----------------------------
# Load everything
# ----------------------------
try:
    model = load_model()
    scaler = load_scaler()
    bounds = load_bounds()
    df = load_data()
except Exception as e:
    st.error(f"Failed to load model/data files: {e}")
    st.stop()

inflation_min = bounds['min']
inflation_max = bounds['max']

# ----------------------------
# Title
# ----------------------------
st.title("Kenya Headline Inflation Rate Forecast")
st.caption("GRU-based multivariate time series forecasting model")

# ----------------------------
# Historical trend chart
# ----------------------------
st.subheader("Historical Inflation Trend")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df['date'], df['headline_inflation'], color='steelblue', linewidth=1.5)
ax1.set_xlabel("Date")
ax1.set_ylabel("Inflation (%)")
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# ----------------------------
# Next month prediction
# ----------------------------
st.subheader("Next Month Forecast")

if len(df) < SEQ_LEN:
    st.warning(f"Need at least {SEQ_LEN} rows of data to forecast; only have {len(df)}.")
else:
    # Scale the full feature set the same way training did
    scaled_values = scaler.transform(df[FEATURE_COLS])

    # Take the last SEQ_LEN months as the input sequence
    last_seq = scaled_values[-SEQ_LEN:]
    x_input = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)  # shape (1, seq_len, features)

    with torch.no_grad():
        pred_scaled = model(x_input).item()

    pred_actual = pred_scaled * (inflation_max - inflation_min) + inflation_min

    last_date = df['date'].max()
    next_date = (last_date + pd.DateOffset(months=1)).strftime('%B %Y')

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"Predicted inflation — {next_date}", value=f"{pred_actual:.2f}%")
    with col2:
        last_actual = df['headline_inflation'].iloc[-1]
        delta = pred_actual - last_actual
        st.metric(label="Change vs last recorded month", value=f"{delta:+.2f} pts")

st.caption("Model: GRU (hidden_size=128, seq_len=12) · Data: KNBS, CBK, EIA/FRED")
