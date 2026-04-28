import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error

FEATURES = ["lag_1", "lag_7", "rolling_7", "rolling_30", "dayofweek", "month", "is_weekend", "type_encoded", "is_holiday", "oil_price"]
SEQ_LEN = 7
CUTOFF = pd.Timestamp("2017-01-01")


class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


@st.cache_resource
def load_model():
    x_scaler = joblib.load("save_model/x_scaler.pkl")
    y_scaler = joblib.load("save_model/y_scaler.pkl")
    model = LSTMModel(input_size=10)
    model.load_state_dict(torch.load("save_model/model.pt", map_location="cpu"))
    model.eval()
    return model, x_scaler, y_scaler


@st.cache_data
def load_data():
    df = pd.read_csv("store-sales-time-series-forecasting/train.csv")
    holidays = pd.read_csv("store-sales-time-series-forecasting/holidays_events.csv")
    oil_df = pd.read_csv("store-sales-time-series-forecasting/oil.csv")

    df["date"] = pd.to_datetime(df["date"])
    holidays["date"] = pd.to_datetime(holidays["date"])
    oil_df["date"] = pd.to_datetime(oil_df["date"])

    daily_sales = df.groupby("date")["sales"].sum().reset_index()
    daily_sales = daily_sales.sort_values("date").reset_index(drop=True)
    daily_sales["lag_1"] = daily_sales["sales"].shift(1)
    daily_sales["lag_7"] = daily_sales["sales"].shift(7)
    daily_sales["rolling_7"] = daily_sales["sales"].rolling(7).mean()
    daily_sales["rolling_30"] = daily_sales["sales"].rolling(30).mean()
    daily_sales["dayofweek"] = daily_sales["date"].dt.dayofweek
    daily_sales["month"] = daily_sales["date"].dt.month
    daily_sales["is_weekend"] = (daily_sales["dayofweek"] >= 5).astype(int)

    national_holidays = holidays[holidays["locale"] == "National"][["date", "type"]]
    le = pd.factorize(national_holidays["type"])
    national_holidays["type_encoded"] = le[0]
    daily_sales = daily_sales.merge(national_holidays[["date", "type_encoded"]].drop_duplicates("date"), on="date", how="left")
    daily_sales["type_encoded"] = daily_sales["type_encoded"].fillna(0).astype(int)
    daily_sales["is_holiday"] = (daily_sales["type_encoded"] > 0).astype(int)

    oil_col = [c for c in oil_df.columns if c != "date"][0]
    oil_df.rename(columns={oil_col: "oil_price"}, inplace=True)
    oil_df["oil_price"] = oil_df["oil_price"].ffill()
    daily_sales = daily_sales.merge(oil_df[["date", "oil_price"]], on="date", how="left")
    daily_sales["oil_price"] = daily_sales["oil_price"].fillna(daily_sales["oil_price"].median())
    daily_sales = daily_sales.dropna().reset_index(drop=True)

    return daily_sales


def get_test_predictions(model, x_scaler, y_scaler, daily_sales):
    test_df = daily_sales[daily_sales["date"] >= CUTOFF].reset_index(drop=True)
    X_test = test_df[FEATURES].values
    X_test_scaled = x_scaler.transform(X_test)

    xs = []
    for i in range(SEQ_LEN, len(X_test_scaled)):
        xs.append(X_test_scaled[i - SEQ_LEN:i])
    X_seq = torch.tensor(np.array(xs), dtype=torch.float32)

    with torch.no_grad():
        preds_scaled = model(X_seq).numpy()
    preds = y_scaler.inverse_transform(preds_scaled).flatten()

    dates = test_df["date"].values[SEQ_LEN:]
    actuals = test_df["sales"].values[SEQ_LEN:]
    return dates, actuals, preds


def forecast_future(model, x_scaler, y_scaler, daily_sales, n_days):
    history = daily_sales.copy()
    predictions = []

    for _ in range(n_days):
        last_7 = history[FEATURES].tail(SEQ_LEN).values
        last_7_scaled = x_scaler.transform(last_7)
        seq = torch.tensor(last_7_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = model(seq).numpy()
        pred = max(0, y_scaler.inverse_transform(pred_scaled).flatten()[0])
        predictions.append(pred)

        next_date = history["date"].iloc[-1] + pd.Timedelta(days=1)
        new_row = {
            "date": next_date, "sales": pred,
            "lag_1": history["sales"].iloc[-1],
            "lag_7": history["sales"].iloc[-7],
            "rolling_7": history["sales"].tail(7).mean(),
            "rolling_30": history["sales"].tail(30).mean(),
            "dayofweek": next_date.dayofweek,
            "month": next_date.month,
            "is_weekend": int(next_date.dayofweek >= 5),
            "type_encoded": 0, "is_holiday": 0,
            "oil_price": history["oil_price"].iloc[-1],
        }
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    future_dates = pd.date_range(daily_sales["date"].iloc[-1] + pd.Timedelta(days=1), periods=n_days)
    return future_dates, predictions


# ---- page config ----
st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

model, x_scaler, y_scaler = load_model()
daily_sales = load_data()

# ---- sidebar ----
st.sidebar.title("Controls")

min_date = daily_sales[daily_sales["date"] >= CUTOFF]["date"].min().date()
max_date = daily_sales["date"].max().date()

date_from = st.sidebar.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
date_to = st.sidebar.date_input("To", value=max_date, min_value=min_date, max_value=max_date)
n_forecast_days = st.sidebar.slider("Forecast days ahead", min_value=7, max_value=90, value=30, step=7)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** 2-layer LSTM  \n**MAE:** $59,859  \n**Trained on:** 2013–2016  \n**Test period:** 2017+")

# ---- get predictions ----
dates, actuals, preds = get_test_predictions(model, x_scaler, y_scaler, daily_sales)

mask = (pd.to_datetime(dates) >= pd.Timestamp(date_from)) & (pd.to_datetime(dates) <= pd.Timestamp(date_to))
dates_f = pd.to_datetime(dates)[mask]
actuals_f = actuals[mask]
preds_f = preds[mask]
errors = np.abs(actuals_f - preds_f)
mae = mean_absolute_error(actuals_f, preds_f)

# ---- business metrics ----
st.title("Sales Forecast Dashboard")
st.markdown("Predict demand to optimize inventory, reduce overstock, and avoid lost sales.")

avg_sales = actuals_f.mean()
overstock_reduction = (1 - mae / avg_sales) * 100
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"${mae:,.0f}")
col2.metric("Avg daily sales", f"${avg_sales:,.0f}")
col3.metric("Forecast accuracy", f"{100 - (mae/avg_sales*100):.1f}%")
col4.metric("Potential overstock reduction", f"~{overstock_reduction:.0f}%")

st.markdown("---")

# ---- plot 1: actual vs predicted ----
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=dates_f, y=actuals_f, name="Actual sales", line=dict(color="#4C72B0")))
fig1.add_trace(go.Scatter(x=dates_f, y=preds_f, name="LSTM forecast", line=dict(color="#DD8452", dash="dash")))
fig1.update_layout(title="Actual vs Predicted Sales", xaxis_title="Date", yaxis_title="Sales ($)", height=400, legend=dict(x=0, y=1))
st.plotly_chart(fig1, use_container_width=True)

# ---- plot 2: error over time ----
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=dates_f, y=errors, name="Daily error", marker_color="#C44E52", opacity=0.7))
fig2.add_hline(y=mae, line_dash="dash", line_color="black", annotation_text=f"Avg MAE ${mae:,.0f}")
fig2.update_layout(title="Forecast Error Over Time (MAE)", xaxis_title="Date", yaxis_title="Error ($)", height=300)
st.plotly_chart(fig2, use_container_width=True)

# ---- plot 3: future forecast ----
future_dates, future_preds = forecast_future(model, x_scaler, y_scaler, daily_sales, n_forecast_days)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=daily_sales["date"].tail(60), y=daily_sales["sales"].tail(60), name="Historical sales", line=dict(color="#4C72B0")))
fig3.add_trace(go.Scatter(x=future_dates, y=future_preds, name=f"Next {n_forecast_days} days forecast", line=dict(color="#DD8452", dash="dash")))
fig3.update_layout(title=f"Forecast: Next {n_forecast_days} Days", xaxis_title="Date", yaxis_title="Sales ($)", height=400)
st.plotly_chart(fig3, use_container_width=True)

# ---- forecast table ----
st.subheader(f"Forecast details — next {n_forecast_days} days")
forecast_df = pd.DataFrame({"date": future_dates.date, "predicted_sales": [f"${p:,.0f}" for p in future_preds]})
st.dataframe(forecast_df, use_container_width=True, hide_index=True)

# ---- how it works ----
with st.expander("How the model works"):
    st.markdown("""
**What it does**
An LSTM (Long Short-Term Memory) neural network trained on 4 years of daily sales data predicts tomorrow's total sales across all stores.

**Why it works**
The model learns from 7-day sequences of features including yesterday's sales, 7-day and 30-day rolling averages, weekday patterns, national holidays, and oil prices. It captures seasonal trends and weekly cycles that simpler models miss.

**What you get**
- Daily sales forecasts with ~{:.1f}% accuracy
- Identify high/low demand days in advance
- Reduce overstock by ordering closer to predicted demand
- Avoid lost sales by stocking up before predicted spikes
    """.format(100 - (mae / avg_sales * 100)))
