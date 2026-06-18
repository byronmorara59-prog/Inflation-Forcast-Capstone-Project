# Kenya Headline Inflation Rate Prediction
### A Multivariate Time Series Forecasting Approach Using Deep Learning on Kenyan Macroeconomic Data

---

| | |
|---|---|
| **Problem Type** | Multivariate Time Series Forecasting |
| **Target Variable** | Kenya Headline CPI Inflation Rate (monthly, one-step ahead) |
| **Models** | LSTM / GRU (Deep Learning) · ARIMA / SARIMA (classical baseline) · XGBoost with lag features (ML baseline) |
| **Output** | Regression (exact rate) + Classification (Up / Down / Stable) |
| **Data Range** | 2009 – 2024 (post-CPI rebase harmonisation) |
| **Data Sources** | KNBS, CBK, World Bank/IMF, EIA, FAO, FRED — no Kaggle datasets |
| **Deployment** | Streamlit |

---

## 1. Target Audience

- **Central Bank of Kenya (CBK)** — anticipate inflation shifts before MPC meetings rather than reacting after CPI is published, enabling more proactive monetary policy
- **National Treasury & Ministry of Finance** — adjust government spending and borrowing decisions based on forward-looking inflation signals for better fiscal planning
- **Commercial Banks & Lenders** — price loans, mortgages, and fixed-income products more accurately by incorporating predicted inflation into risk models
- **Investment Firms & Fund Managers** — improve inflation-adjusted return forecasting for bonds, equities, and money market funds listed on the NSE
- **Import/Export Businesses** — forward-price contracts and hedge currency exposure based on anticipated cost-of-living and exchange rate pressures
- **SMEs & Retailers** — anticipate input cost increases and adjust pricing strategy in advance rather than absorbing unexpected margin compression
- **Economic Researchers & Policy Analysts** — a benchmark for ML-based macroeconomic forecasting in Sub-Saharan Africa, with methodology transferable to other economies

---

## 2. Project Overview

This project develops a machine learning system to forecast Kenya's monthly headline inflation rate using historical macroeconomic time series data. The system ingests publicly available economic indicators — sourced via web scraping from Kenyan and international institutions — and trains deep learning models to predict the next month's Consumer Price Index (CPI) inflation figure.

The project is structured around three pillars:

- **Data engineering** — scraping, cleaning, and harmonising multi-source macroeconomic data into a unified monthly dataset
- **Modelling** — training and benchmarking LSTM/GRU models against classical (ARIMA) and ML (XGBoost) baselines
- **Deployment** — delivering predictions via streamlit

---

## 3. Problem Statement

Inflation is one of the most consequential macroeconomic indicators in Kenya, directly affecting household purchasing power, business pricing decisions, loan affordability, and government fiscal policy. Despite its importance, inflation forecasting in Kenya remains largely reactive — businesses, policymakers, and households respond to published figures after they are released by the Kenya National Bureau of Statistics (KNBS) rather than anticipating changes in advance.

This reactive posture has tangible consequences:

- Businesses cannot accurately price goods and contracts months ahead, leading to margin compression or over-pricing
- The Central Bank of Kenya (CBK) adjusts monetary policy after inflation has already shifted, reducing proactive effectiveness
- Households receive no advance signal of cost-of-living increases
- Investors and lenders cannot accurately price inflation risk into long-term financial agreements

---

## 4. Target Variable — Headline Inflation

Kenya publishes two primary inflation measures. This project targets headline inflation for the reasons outlined below:

| Measure | Description |
|---|---|
| **Headline Inflation (Selected)** | The overall CPI — includes all goods and services in the basket: food, fuel, housing, education, health, etc. This is the figure published monthly by KNBS and widely reported in public discourse. Selected for this project due to its public relevance, data availability, and direct policy and business impact. |
| **Core Inflation (Future Extension)** | Headline CPI minus food and fuel components. Reflects underlying structural inflation trends, stripping out volatile seasonal components. Noted as a potential extension once the headline model is validated. |

---

## 5. Data Harmonisation & Study Period

A critical research decision is ensuring that inflation data is methodologically consistent across the entire study period. Kenya's CPI measurement has undergone significant revisions over time, making pre-rebase data incomparable with post-rebase data.

| Period | CPI Methodology | Research Implication |
|---|---|---|
| Pre-2009 | Based on outdated 1997 Household Budget Survey weights | Not comparable with modern data — excluded from study |
| 2009 onward | Rebased using 2005/2006 Household Budget Survey — major structural overhaul of basket weights | **Study Start Point** — consistent, modern methodology |

---

## 6. Dataset Description


### 6.1 Feature Categories

Rather than tracking individual commodity prices, KNBS publishes pre-aggregated CPI sub-indices that combine all items within each category. These are used directly as features.

**CPI Sub-Indices (from KNBS)**

| Sub-Index | What It Captures | Approx. CPI Weight |
|---|---|---|
| Food & Non-Alcoholic Beverages | Maize, rice, bread, milk, vegetables — largest CPI component | ~36% |
| Transport Index | Fuel pump prices, matatu fares, transport costs | ~8% |
| Housing, Water, Electricity & Gas | Rent, electricity bills, cooking gas | ~18% |
| Health Index | Medicine, hospital and clinic fees | ~3% |
| Education Index | School and college tuition fees | ~5% |

**Monetary & Macroeconomic Indicators**

| Indicator | Description |
|---|---|
| Money Supply (M3) | Broad money in circulation — excess supply drives inflation (CBK) |
| CBK Central Bank Rate | Benchmark lending rate — higher rate cools inflation (CBK) |
| KES/USD Exchange Rate | Weak shilling increases import costs → imported inflation (CBK) |
| GDP Growth Rate | Higher growth can trigger demand-pull inflation (World Bank/IMF) |
| Government Fiscal Deficit | Excess spending injects money into economy, fuelling inflation (World Bank) |
| Credit Growth | How aggressively banks are lending — demand stimulus signal (CBK) |

**Global / External Indicators**

| Indicator | Description |
|---|---|
| Brent Crude Oil Price | Global oil price feeds directly into Kenya's fuel and transport sub-index (EIA) |
| FAO Food Price Index | Global commodity prices for maize, wheat — key imports affecting food sub-index (FAO) |
| US Federal Reserve Rate | Global monetary conditions influence capital flows and KES strength (FRED) |

---

## 7. Data Sourcing Plan


| Source | Data Provided | Collection Method |
|---|---|---|
| KNBS (knbs.or.ke) | Monthly CPI, all sub-indices, headline inflation figures | BeautifulSoup or selenium scraper + Excel/PDF report parsing |
| CBK (cbk.go.ke) | CBR history, money supply (M3), KES/USD rates, forex reserves, credit growth | BeautifulSoup or selenium + pdfplumber for PDF reports |
| World Bank / IMF API | GDP growth, government spending, fiscal deficit, trade balance | Public JSON REST API |
| EIA (eia.gov) | Global Brent crude oil prices — monthly averages | EIA Open Data API / CSV download |
| FAO (fao.org) | Global food commodity price index — monthly | FAO Data API / CSV download |
| FRED (Federal Reserve) | US Fed funds rate, DXY dollar index | fredapi Python library |



## 8. Methodology

### 8.1 Problem Framing

| | |
|---|---|
| **Time Series Forecasting** | The fundamental nature of the problem — data is sequential, monthly, and order-dependent |
| **Regression Task** | Predict the exact headline CPI inflation rate (%) for next month — for example output: 6.3% |
| **Classification Task** | Predict the direction of change: Will inflation Rise / Fall / Stay Stable next month? |
| **Forecast Horizon** | One-step ahead (next month) |

### 8.2 Models

| Model | Type | Role & Rationale |
|---|---|---|
| LSTM (Long Short-Term Memory) | Deep Learning — Recurrent | **Primary:** Designed for sequential data; captures long-range dependencies across months |
| GRU (Gated Recurrent Unit) | Deep Learning — Recurrent | **Secondary:** Lighter, faster alternative to LSTM with fewer parameters — compared for performance vs. efficiency. |
| ARIMA / SARIMA | Classical Time Series | **Baseline:** SARIMA variant handles Kenya's seasonal inflation patterns |
| XGBoost with Lag Features | Gradient Boosting (ML) | **ML Baseline:** Strong non-deep-learning benchmark on engineered lag features |

### 8.3 Key Technical Considerations

| Consideration | Detail |
|---|---|
| Lag Features | Inflation this month is influenced by indicators from prior months. A lag window (e.g. 3–6 months back) is engineered as additional input features |
| Seasonality | Kenya's inflation has seasonal patterns — food prices spike during dry seasons. SARIMA models this explicitly; month-of-year added as a cyclical feature for LSTM/GRU |
| Stationarity | Augmented Dickey-Fuller (ADF) test applied; differencing used where necessary before model training |
| Missing Values | Gaps handled via forward fill or linear interpolation — strategy documented per variable |
| Frequency Alignment | Quarterly indicators (e.g. GDP) interpolated to monthly frequency to match CPI cadence |
| Feature Standardisation | All features standardised (zero mean, unit variance) before training to ensure numerical stability |

---

## 9. Train / Validation / Test Split

| Split | Period |
|---|---|
| **Training Set** | January 2009 – December 2021 (~156 months) |
| **Validation Set** | January 2022 – December 2023 (24 months) — for hyperparameter tuning |
| **Test Set** | January 2024 – December 2024 (12 months) |
| **Validation Method** | Walk-forward (rolling) validation|

---

## 10. Evaluation Metrics

| Metric | Task | What It Measures |
|---|---|---|
| MAE (Mean Absolute Error) | Regression | Average absolute difference between predicted and actual rate — most interpretable for non-technical audiences |
| RMSE (Root Mean Squared Error) | Regression | Penalises large prediction errors more heavily — useful for catching months with big misses |
| MAPE (Mean Abs. % Error) | Regression | % error relative to actual value — easy to communicate business impact |

---

## 11. Expected Deliverables

1. Structured, harmonised dataset — unified monthly CSV (2009–2024) built from KNBS, CBK, World Bank, EIA, FAO, and FRED
2. Trained LSTM model predicting Kenya's monthly headline inflation rate — primary deliverable
3. Trained GRU model — architecture comparison against LSTM
4. ARIMA / SARIMA baseline — classical time series benchmark
5. XGBoost baseline — gradient boosting benchmark with lag features
6. Model comparison report: all four models evaluated on MAE, RMSE, and MAPE
7. Backtesting results — predicted vs. actual inflation for 2024 test period with real KNBS verification
8. Feature importance analysis — identifying which macroeconomic indicators most drive Kenya's inflation
9. Deployed web application — live monthly forecasting with visualisation dashboard
10. Full reproducible pipeline: data scraper → preprocessor → feature engineer → model trainer → deployed app

---

## 12. Limitations & Future Work

### 12.1 Known Limitations

- ~180 monthly observations is a relatively small dataset for deep learning — regularisation and careful cross-validation are critical
- Some KNBS sub-index data before 2010 may have gaps requiring interpolation — documented as a data quality limitation
- Structural breaks (COVID-19 2020, post-election shocks) may distort model performance in those periods
- LSTM/GRU models are sensitive to hyperparameter choices — systematic tuning via grid search required

### 12.2 Future Work

- Extend to core inflation prediction and compare performance between headline and core models
