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
    div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
        background-color: {BG};
        color: {TEXT};
        border: 1px solid {CARD_BORDER};
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
        padding: 22px 24px;
        margin-top: 8px;
        margin-bottom: 18px;
    }}
    hr {{
        border-color: {CARD_BORDER} !important;
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

MONTH_NAMES = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']

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
    df['m3_pct_change'] = 0.0  # Forcefully zeroing historical tracking out
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

# ============================================================
# Header
# ============================================================
st.markdown("## Kenya Headline Inflation Rate Forecast")
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
# Next month forecast (automatic, using the real last 12 months)
# ============================================================
st.markdown("#### Next Month Forecast")

if len(df) < SEQ_LEN:
    st.warning(f"Need at least {SEQ_LEN} rows of data to forecast; only have {len(df)}.")
else:
    scaled_values = scaler.transform(df[FEATURE_COLS])
    last_seq = scaled_values[-SEQ_LEN:]
    x_input = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(x_input).item()
    pred_actual = pred_scaled * (inflation_max - inflation_min) + inflation_min

    last_date = df['date'].max()
    next_date = last_date + pd.DateOffset(months=1)
    next_month_name = next_date.strftime('%B %Y')
    last_actual = df['headline_inflation'].iloc[-1]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"Predicted inflation — {next_month_name}", value=f"{pred_actual:.2f}%")
    with col2:
        delta = pred_actual - last_actual
        st.metric(label="Change vs last recorded month", value=f"{delta:+.2f} pts")

st.write("")

# ============================================================
# Custom Inflation Prediction — manual input calculator
# ============================================================
st.markdown("---")
st.markdown("#### 🔮 Custom Inflation Prediction")
st.markdown("Enter current economic indicators to predict headline inflation for a chosen month:")

st.markdown('<div class="section-card">', unsafe_allow_html=True)

last_row = df.iloc[-1]

c1, c2, c3 = st.columns(3)
with c1:
    input_headline = st.number_input(
        "Current Month's Headline Inflation (%)", min_value=0.0, max_value=30.0,
        value=float(last_row['headline_inflation']), step=0.1
    )
    input_brent = st.number_input(
        "Brent Crude Oil (USD/barrel)", min_value=20.0, max_value=200.0,
        value=float(last_row['brent_crude']), step=0.5
    )
    input_cbk = st.number_input(
        "CBK Rate (%)", min_value=1.0, max_value=25.0,
        value=float(last_row['cbk_rate']), step=0.25
    )
with c2:
    input_kes = st.number_input(
        "KES/USD Exchange Rate", min_value=50.0, max_value=250.0,
        value=float(last_row['kes_usd']), step=0.5
    )
    input_m3 = st.number_input(
        "Money Supply M3 (KSh Millions)", min_value=500000.0, max_value=15000000.0,
        value=float(last_row['m3']), step=10000.0
    )
    input_forex = st.number_input(
        "Forex Reserves (KSh Millions)", min_value=50000.0, max_value=2000000.0,
        value=float(last_row['forex_reserves']), step=5000.0
    )
with c3:
    # m3_pct_change user input box removed here to clean UI layout
    input_month = st.selectbox(
        "Month These Indicators Describe", options=list(range(1, 13)),
        format_func=lambda x: MONTH_NAMES[x - 1],
        index=int(last_row['month']) - 1
    )

st.markdown('</div>', unsafe_allow_html=True)

if st.button("Predict Inflation", type="primary"):
    # Generate custom dataframe row structure matching the scaler vector shapes
    custom_data = pd.DataFrame([{
        'headline_inflation': input_headline,
        'brent_crude': input_brent,
        'cbk_rate': input_cbk,
        'kes_usd': input_kes,
        'm3': input_m3,
        'forex_reserves': input_forex,
        'month': input_month,
        'm3_pct_change': 0.0  # Safely hardcoding neutral state vector parameter
    }])
    
    # Process custom scale configurations
    try:
        custom_scaled = scaler.transform(custom_data[FEATURE_COLS])
        
        # Simulating dummy seq matrix matching GRU structure layout expectations
        dummy_seq = np.repeat(custom_scaled, SEQ_LEN, axis=0).reshape(1, SEQ_LEN, -1)
        x_custom_input = torch.tensor(dummy_seq, dtype=torch.float32)
        
        with torch.no_grad():
            custom_pred_scaled = model(x_custom_input).item()
            
        custom_pred_actual = custom_pred_scaled * (inflation_max - inflation_min) + inflation_min
        custom_delta = custom_pred_actual - input_headline
        
        st.write("")
        st.markdown("#### Custom Prediction Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Predicted Inflation", value=f"{custom_pred_actual:.2f}%")
        with res_col2:
            st.metric(label="Change vs Input Month", value=f"{custom_delta:+.2f} pts")
    except Exception as calc_err:
        st.error(f"Prediction matrix matching error: {calc_err}")

st.markdown('<div class="footer-credit">Kenya Inflation Forecast Model Workspace</div>', unsafe_allow_html=True)
