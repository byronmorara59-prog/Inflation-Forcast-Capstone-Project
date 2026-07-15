import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kenya Inflation Forecast", layout="wide", page_icon="📈")

# ============================================================
# DARK THEME
# ============================================================
BG = "#0F1420"
CARD_BG = "#171D2B"
CARD_BORDER = "#262E42"
GOLD = "#C8901E"
RUST = "#B14226"
TEXT = "#E8EAED"
MUTED = "#93A0B4"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {CARD_BG};
    }}
    h1, h2, h3, h4, p, span, label, div {{
        color: {TEXT};
    }}
    .subtitle {{
        color: {MUTED};
        font-size: 15px;
        margin-top: -8px;
    }}
    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-left: 4px solid {GOLD};
        border-radius: 10px;
        padding: 16px 18px;
    }}
    div[data-testid="stMetric"] label {{
        color: {MUTED} !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {TEXT} !important;
    }}
    .block-container {{
        padding-top: 2.2rem;
    }}
    .footer-credit {{
        text-align: center;
        color: {MUTED};
        font-size: 13px;
        margin-top: 2.5rem;
        border-top: 1px solid {CARD_BORDER};
        padding-top: 14px;
    }}
    .section-card {{
        background-color: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 18px;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Model definition (must match training exactly)
# ============================================================
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
TARGET_IDX = FEATURE_COLS.index('headline_inflation')
MONTH_IDX = FEATURE_COLS.index('month')
FORECAST_MONTHS = 6

# ============================================================
# Cached loaders
# ============================================================
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
month_min = scaler.data_min_[MONTH_IDX]
month_max = scaler.data_max_[MONTH_IDX]

# ============================================================
# Header
# ============================================================
st.markdown("## 🇰🇪 Kenya Headline Inflation Rate Forecast")
st.markdown('<p class="subtitle">GRU-based multivariate time series forecasting model</p>', unsafe_allow_html=True)
st.write("")

# ============================================================
# Historical trend chart (dark themed)
# ============================================================
st.markdown("#### Historical Inflation Trend")
fig1, ax1 = plt.subplots(figsize=(12, 4))
fig1.patch.set_facecolor(BG)
ax1.set_facecolor(BG)
ax1.plot(df['date'], df['headline_inflation'], color=GOLD, linewidth=1.8)
ax1.set_xlabel("Date", color=MUTED)
ax1.set_ylabel("Inflation (%)", color=MUTED)
ax1.tick_params(colors=MUTED)
for spine in ax1.spines.values():
    spine.set_color(CARD_BORDER)
ax1.grid(True, alpha=0.15, color=MUTED)
st.pyplot(fig1)

st.write("")

# ============================================================
# Recursive multi-month forecast
# ============================================================
st.markdown(f"#### {FORECAST_MONTHS}-Month Forecast")

if len(df) < SEQ_LEN:
    st.warning(f"Need at least {SEQ_LEN} rows of data to forecast; only have {len(df)}.")
else:
    scaled_values = scaler.transform(df[FEATURE_COLS])
    window = scaled_values[-SEQ_LEN:].copy()

    last_date = df['date'].max()
    forecast_dates, forecast_values = [], []

    for step in range(FORECAST_MONTHS):
        x_input = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(x_input).item()
        pred_actual = pred_scaled * (inflation_max - inflation_min) + inflation_min

        next_date = last_date + pd.DateOffset(months=step + 1)
        forecast_dates.append(next_date)
        forecast_values.append(pred_actual)

        # Build the next row: carry forward the last known exogenous features,
        # feed the prediction back in as the new "headline_inflation", and advance the month.
        new_row = window[-1].copy()
        new_row[TARGET_IDX] = pred_scaled
        next_month_scaled = ((next_date.month) - month_min) / (month_max - month_min)
        new_row[MONTH_IDX] = next_month_scaled

        window = np.vstack([window[1:], new_row])

    col1, col2 = st.columns([1.3, 1])

    with col1:
        fig2, ax2 = plt.subplots(figsize=(7, 3.6))
        fig2.patch.set_facecolor(BG)
        ax2.set_facecolor(BG)
        recent_hist = df.tail(12)
        ax2.plot(recent_hist['date'], recent_hist['headline_inflation'], color=MUTED, linewidth=1.6, label="Actual")
        ax2.plot(forecast_dates, forecast_values, color=RUST, linewidth=2, linestyle="--", marker="o", label="Forecast")
        ax2.axvline(last_date, color=CARD_BORDER, linestyle=":", linewidth=1)
        ax2.set_ylabel("Inflation (%)", color=MUTED)
        ax2.tick_params(colors=MUTED, labelsize=8)
        ax2.legend(facecolor=CARD_BG, edgecolor=CARD_BORDER, labelcolor=TEXT, fontsize=9)
        for spine in ax2.spines.values():
            spine.set_color(CARD_BORDER)
        ax2.grid(True, alpha=0.15, color=MUTED)
        st.pyplot(fig2)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        for d, v in zip(forecast_dates, forecast_values):
            st.metric(label=d.strftime('%B %Y'), value=f"{v:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        "⚠️ Only the next-month forecast is validated (test MAPE 14.87%). Months 2–6 are an illustrative "
        "recursive projection: the model's own prediction is fed back in as history, while oil prices, the "
        "CBK rate, the exchange rate, money supply and forex reserves are held at their last known values. "
        "Treat this as a rough trend indicator beyond month 1, not a validated forecast."
    )

st.markdown(
    '<div class="footer-credit">Model: GRU (hidden_size=128, seq_len=12) · Data: KNBS, CBK, EIA/FRED</div>',
    unsafe_allow_html=True
)