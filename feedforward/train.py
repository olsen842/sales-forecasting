import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---- load and prep data ----

df = pd.read_csv("/Users/eliasolsen/Documents/programering/mappe uden navn/store-sales-time-series-forecasting/train.csv")
holidays = pd.read_csv("/Users/eliasolsen/Documents/programering/mappe uden navn/store-sales-time-series-forecasting/holidays_events.csv")
oil_df = pd.read_csv("/Users/eliasolsen/Documents/programering/mappe uden navn/store-sales-time-series-forecasting/oil.csv")
df["date"] = pd.to_datetime(df["date"])
holidays["date"] = pd.to_datetime(holidays["date"])
oil_df["date"] = pd.to_datetime(oil_df["date"])





# sum all stores and families into one daily number
daily_sales = df.groupby("date")["sales"].sum().reset_index()
daily_sales = daily_sales.sort_values("date").reset_index(drop=True)

# lag features - yesterday and same day last week
daily_sales["lag_1"] = daily_sales["sales"].shift(1)
daily_sales["lag_7"] = daily_sales["sales"].shift(7)

# rolling averages
daily_sales["rolling_7"] = daily_sales["sales"].rolling(7).mean()
daily_sales["rolling_30"] = daily_sales["sales"].rolling(30).mean()

# calendar stuff
daily_sales["dayofweek"] = daily_sales["date"].dt.dayofweek
daily_sales["month"] = daily_sales["date"].dt.month
daily_sales["is_weekend"] = (daily_sales["dayofweek"] >= 5).astype(int)

#holiday features
national_holidays = holidays[holidays['locale'] == 'National'][['date', 'type']]

le = pd.factorize(national_holidays['type'])
national_holidays['type_encoded'] = le[0]

daily_sales = daily_sales.merge(national_holidays[['date', 'type_encoded']].drop_duplicates('date'), on='date', how='left')
daily_sales['type_encoded'] = daily_sales['type_encoded'].fillna(0).astype(int)
daily_sales['is_holiday'] = (daily_sales['type_encoded'] > 0).astype(int)

#oli feature
oil_col = [c for c in oil_df.columns if c != 'date'][0] 
oil_df.rename(columns={oil_col: 'oil_price'}, inplace=True)
oil_df['oil_price'] = oil_df['oil_price'].ffill()
daily_sales = daily_sales.merge(oil_df[['date', 'oil_price']], on='date', how='left')
daily_sales['oil_price'] = daily_sales['oil_price'].fillna(daily_sales['oil_price'].median())


# first 30 rows have NaN from rolling_30
daily_sales = daily_sales.dropna().reset_index(drop=True)

print(f"loaded {len(daily_sales)} days")
print(f"date range: {daily_sales['date'].min().date()} to {daily_sales['date'].max().date()}")


# ---- train/test split ----
FEATURES = [
    "lag_1", "lag_7", "rolling_7", "rolling_30", 
    "dayofweek", "month", "is_weekend",
    "type_encoded", "is_holiday", "oil_price"
]
cutoff = pd.Timestamp("2017-01-01")

train_df = daily_sales[daily_sales["date"] < cutoff]
test_df = daily_sales[daily_sales["date"] >= cutoff]

X_train = train_df[FEATURES]
y_train = train_df["sales"]
X_test = test_df[FEATURES]
y_test = test_df["sales"]

# scale features
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# scale target too - without this training didn't converge at all
# first attempt gave MAE of $873k which was worse than predicting zero
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

print(f"train: {len(train_df)} days | test: {len(test_df)} days")


# ---- baselines ----

# baseline 1: just predict yesterday's sales
pred_bl1 = test_df["lag_1"].values
mae_bl1 = mean_absolute_error(y_test, pred_bl1)
rmse_bl1 = np.sqrt(mean_squared_error(y_test, pred_bl1))

# baseline 2: 7-day rolling average
pred_bl2 = test_df["rolling_7"].values
mae_bl2 = mean_absolute_error(y_test, pred_bl2)
rmse_bl2 = np.sqrt(mean_squared_error(y_test, pred_bl2))

print(f"\nbaseline results:")
print(f"  yesterday:     MAE=${mae_bl1:,.0f}  RMSE=${rmse_bl1:,.0f}")
print(f"  7-day avg:     MAE=${mae_bl2:,.0f}  RMSE=${rmse_bl2:,.0f}")


# ---- neural network ----

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# 32 -> 16 -> 1, no sigmoid on output because this is regression
model = nn.Sequential(
    nn.Linear(len(FEATURES), 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 350
print(f"\ntraining for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test_tensor), y_test_tensor).item()
        print(f"  epoch {epoch+1}/{epochs} | train: {loss.item():.4f} | test: {test_loss:.4f}")

# get predictions back in dollar amounts
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()
y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

mae_nn = mean_absolute_error(y_test, y_pred)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nneural network: MAE=${mae_nn:,.0f}  RMSE=${rmse_nn:,.0f}")

# save everything evaluate.py needs
torch.save(model.state_dict(), "../results/model.pt")
np.save("../results/y_pred.npy", y_pred)
np.save("../results/y_test.npy", y_test.values)
np.save("../results/test_dates.npy", test_df["date"].values)
np.save("../results/baseline_pred.npy", pred_bl2)
print("saved model and predictions to results/")
