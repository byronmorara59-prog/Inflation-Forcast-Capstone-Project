# Kenya Headline Inflation Rate Prediction
### A Multivariate Time Series Forecasting Approach Using Deep Learning on Kenyan Macroeconomic Data

---

| | |
|---|---|
| **Problem Type** | Multivariate Time Series Forecasting |
| **Target Variable** | Kenya Headline CPI Inflation Rate (monthly, one-step ahead) |
| **Models** | GRU (primary) · LSTM · SARIMA (classical baseline) · XGBoost (ML baseline) |
| **Best Model** | GRU — MAE: 0.66 · RMSE: 0.94 · MAPE: 14.87% |
| **Output** | Regression (exact rate) + Classification (Up / Down / Stable) |
| **Data Range** | 2009 – 2026 (post-CPI rebase harmonisation) |
| **Data Sources** | KNBS, CBK, EIA/FRED — no Kaggle datasets |
| **Deployment** | Streamlit web application |

---

## 1. Target Audience

- **Central Bank of Kenya (CBK)** — anticipate inflation shifts before MPC meetings, enabling more proactive monetary policy
- **National Treasury & Ministry of Finance** — adjust spending and borrowing decisions based on forward-looking inflation signals
- **Commercial Banks & Lenders** — price loans, mortgages, and fixed-income products more accurately
- **Investment Firms & Fund Managers** — improve inflation-adjusted return forecasting for NSE instruments
- **Import/Export Businesses** — forward-price contracts and hedge currency exposure
- **SMEs & Retailers** — anticipate input cost increases and adjust pricing strategy in advance
- **Economic Researchers & Policy Analysts** — benchmark for ML-based macroeconomic forecasting in Sub-Saharan Africa

---

## 2. Project Overview

This project develops a machine learning system to forecast Kenya's monthly headline inflation rate using historical macroeconomic time series data. The system ingests publicly available economic indicators and trains deep learning models to predict the next month's CPI inflation figure.

The project is structured around three pillars:

- **Data engineering** — scraping, cleaning, and harmonising multi-source macroeconomic data into a unified monthly dataset
- **Modelling** — training and benchmarking GRU/LSTM deep learning models against SARIMA and XGBoost baselines
- **Deployment** — delivering predictions via a live Streamlit web application

---

## 3. Problem Statement

Inflation forecasting in Kenya remains largely reactive — businesses, policymakers, and households respond to published figures after they are released by KNBS rather than anticipating changes in advance. This has tangible consequences:

- Businesses cannot accurately price goods and contracts months ahead
- The CBK adjusts monetary policy after inflation has already shifted
- Households receive no advance signal of cost-of-living increases
- Investors cannot accurately price inflation risk into long-term agreements

---

## 4. Target Variable — Headline Inflation

| Measure | Description |
|---|---|
| **Headline Inflation (Selected)** | Overall CPI — includes all goods and services. Published monthly by KNBS. Selected for public relevance and direct policy impact. |
| **Core Inflation (Future Extension)** | Headline CPI minus food and fuel. Noted as a potential future extension. |

---

## 5. Data Harmonisation & Study Period

| Period | CPI Methodology | Research Implication |
|---|---|---|
| Pre-2009 | Based on outdated 1997 Household Budget Survey | Not comparable — excluded |
| 2009 onward | Rebased using 2005/2006 survey — major overhaul | ✔ **Study Start Point** |
| 2019/2020+ | Rebased using 2015/2016 survey — current basket | Refinement within same framework |

> Dataset starts from 2009 to ensure CPI methodology consistency across the full study period.

---

## 6. Dataset Description

### 6.1 Structure

| | |
|---|---|
| **Unit of Observation** | One calendar month |
| **Frequency** | Monthly |
| **Study Period** | January 2009 – February 2026 (~200 observations after feature engineering) |
| **Target Variable (y)** | Kenya Headline CPI Inflation Rate (%) |
| **Features (X)** | brent_crude, cbk_rate, kes_usd, m3_pct_change, forex_reserves, month |
| **Format** | Single merged CSV — date-indexed |

### 6.2 Feature Categories

| Indicator | Description |
|---|---|
| Brent Crude Oil Price | Global oil price feeds into Kenya's fuel and transport costs (EIA/FRED) |
| CBK Central Bank Rate | Benchmark lending rate — higher rate cools inflation (CBK) |
| KES/USD Exchange Rate | Weak shilling increases import costs → imported inflation (CBK) |
| Money Supply M3 (% change) | Monthly rate of change in broad money — captures inflationary pressure (CBK) |
| Forex Reserves | CBK foreign asset holdings — proxy for economic stability (CBK) |
| Month (1-12) | Captures confirmed 12-month seasonal cycle |

---

## 7. Data Sourcing Plan

> All data collected via web scraping and public APIs. No Kaggle datasets used.

| Source | Data Provided | Collection Method |
|---|---|---|
| KNBS (knbs.or.ke) | Monthly headline inflation figures | CSV download from CBK statistics portal |
| CBK (cbk.go.ke) | CBR history, M3, KES/USD, forex reserves | CSV downloads from CBK statistical bulletins |
| EIA / FRED | Brent crude oil prices — monthly averages | FRED direct CSV download (DCOILBRENTEU) |

---

## 8. Data Storage

All scraped data is stored in a **SQLite database** before preprocessing and modelling.

| Table | Contents | Updated |
|---|---|---|
| raw_cbk | CBR, M3, KES/USD, forex reserves | Monthly |
| raw_inflation | Headline inflation from KNBS/CBK | Monthly |
| raw_brent | Brent crude from FRED | Monthly |
| processed_features | Cleaned, scaled features for modelling | On pipeline run |
| predictions | Predicted vs actual inflation per month | On model run |

---

## 9. Methodology

### 9.1 Problem Framing

| | |
|---|---|
| **Time Series Forecasting** | Sequential, monthly, order-dependent data |
| **Regression Task** | Predict exact inflation rate (%) for next month |
| **Classification Task** | Predict direction: Rise / Fall / Stay Stable |
| **Forecast Horizon** | One-step ahead (next month) |

### 9.2 Preprocessing Steps

| Step | Finding / Decision |
|---|---|
| Decomposition | Confirmed long-term downward trend and strong 12-month seasonal cycle |
| Stationarity (ADF) | Borderline stationary at 5% (p=0.048) — no differencing applied |
| ACF/PACF | Significant autocorrelations at lags 1-6 — informed SARIMA configuration |
| Scaling | MinMaxScaler — bounded 0-1 range optimal for GRU/LSTM activations |

### 9.3 Feature Engineering

An ablation study compared models with and without lag features. Lag features consistently degraded performance across all models — attributed to the curse of dimensionality with ~150 training observations. Final models use:

- **M3 percentage change** — monthly rate of change more informative than raw level
- **Month feature** — captures 12-month seasonal cycle
- **No lag features** — no-lag versions outperformed lag versions for all models

### 9.4 Models

| Model | Type | Role |
|---|---|---|
| SARIMA(1,0,1)(1,0,1,12) | Classical Time Series | Baseline — univariate, inflation history only |
| XGBoost | Gradient Boosting | ML Baseline — multivariate, no sequential memory |
| LSTM (seq=12, hidden=128) | Deep Learning — Recurrent | Primary — sequence window of 12 months |
| GRU (seq=12, hidden=128) | Deep Learning — Recurrent | Primary — simpler 2-gate architecture, best performer |

---

## 10. Train / Validation / Test Split

| Split | Period |
|---|---|
| **Training Set** | January 2009 – December 2021 (~150 months) |
| **Validation Set** | January 2022 – December 2023 (24 months) |
| **Test Set** | January 2024 – February 2026 (26 months) — holdout, verified against real KNBS figures |

> Walk-forward validation used throughout — temporal order preserved, no random shuffling.

---

## 11. Model Results

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| SARIMA(1,0,1)(1,0,1,12) | 1.2194 | 1.3854 | 32.87% |
| XGBoost (no lags) | 1.4199 | 1.6015 | 38.26% |
| XGBoost (with lags) | 1.4509 | 1.6277 | 38.94% |
| LSTM (seq=12, hidden=128) | 1.0521 | 1.2951 | 25.83% |
| **GRU (seq=12, hidden=128)** | **0.6619** | **0.9396** | **14.87% ✔** |

**Key findings:**
- GRU is the best performing model — MAPE of 14.87% (good range: 10-20%)
- Deep learning models significantly outperform SARIMA and XGBoost
- GRU outperforms LSTM — simpler architecture generalises better on small datasets
- Lag features degraded performance across all models — curse of dimensionality confirmed
- Sequence length of 12 months optimal — aligns with confirmed seasonal cycle

---

## 12. Evaluation Metrics

| Metric | What It Measures |
|---|---|
| MAE (Mean Absolute Error) | Average absolute difference in percentage points |
| RMSE (Root Mean Squared Error) | Penalises large errors more heavily |
| MAPE (Mean Abs. % Error) | % error relative to actual — primary metric |

---

## 13. Deployment

- **Platform:** Streamlit web application
- **Hosting:** Streamlit Community Cloud
- **Model:** GRU (seq=12, hidden=128) — saved as `gru_model.pth`
- **Update Cadence:** Monthly

**Application Features:**
- Historical inflation trend visualisation (2009 – present)
- Predicted vs actual inflation on the 26-month test set
- Next month inflation forecast
- Model comparison dashboard

---

## 14. Expected Deliverables

1. Structured, harmonised dataset — unified monthly CSV (2009–2026)
2. SQLite database — raw data, processed features, and predictions
3. Trained GRU model (`gru_model.pth`)
4. Trained LSTM model
5. SARIMA baseline
6. XGBoost baseline
7. Ablation study results — lag vs no-lag comparison
8. Model comparison report
9. Backtesting results — 2024-2026 predictions vs real KNBS figures
10. Deployed Streamlit web application
11. Full reproducible pipeline

---

## 15. Limitations & Future Work

### 15.1 Known Limitations

- ~150 training observations is small for deep learning
- Lag features degraded performance — future work could explore PCA before adding lags
- Structural breaks (COVID-19, post-election shocks) affect model generalisation
- All models struggled with the rapid 2024 inflation decline

### 15.2 Future Work

- Extend to core inflation prediction and compare with headline model