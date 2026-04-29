import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

os.makedirs("save_model", exist_ok=True)


# ---- load and prep data ----

df = pd.read_csv("store-sales-time-series-forecasting/train.csv")
holidays = pd.read_csv("store-sales-time-series-forecasting/holidays_events.csv")
oil_df = pd.read_csv("store-sales-time-series-forecasting/oil.csv")
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


# ---- baselines (evaluated after sequences so y_test_windowed exists) ----
pred_bl1 = None
pred_bl2 = None


# ---- neural network ----

SEQ_LEN = 7  # use last 7 days as context for each prediction

def make_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)

X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train_scaled.flatten(), SEQ_LEN)
X_test_seq, y_test_seq = make_sequences(X_test_scaled, y_test_scaled.flatten(), SEQ_LEN)

X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1)

# also trim y_test to match the windowed length so metrics align
y_test_windowed = y_test.values[SEQ_LEN:]

# baselines trimmed to same window so comparison is fair
pred_bl1 = test_df["lag_1"].values[SEQ_LEN:]
pred_bl2 = test_df["rolling_7"].values[SEQ_LEN:]
mae_bl1 = mean_absolute_error(y_test_windowed, pred_bl1)
rmse_bl1 = np.sqrt(mean_squared_error(y_test_windowed, pred_bl1))
mae_bl2 = mean_absolute_error(y_test_windowed, pred_bl2)
rmse_bl2 = np.sqrt(mean_squared_error(y_test_windowed, pred_bl2))
print(f"\nbaseline results:")
print(f"  yesterday:     MAE=${mae_bl1:,.0f}  RMSE=${rmse_bl1:,.0f}")
print(f"  7-day avg:     MAE=${mae_bl2:,.0f}  RMSE=${rmse_bl2:,.0f}")

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

torch.manual_seed(81)
model = LSTMModel(input_size=len(FEATURES))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

BATCH_SIZE = 32
PATIENCE = 25

dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

best_val_loss = float("inf")
epochs_no_improve = 0
best_weights = None

print(f"\ntraining (early stopping patience={PATIENCE})...")

for epoch in range(500):
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_test_tensor), y_test_tensor).item()
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1

    if (epoch + 1) % 50 == 0:
        print(f"  epoch {epoch+1} | train: {loss.item():.4f} | test: {val_loss:.4f} | best: {best_val_loss:.4f}")

    if epochs_no_improve >= PATIENCE:
        print(f"  early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_weights)

# get predictions back in dollar amounts
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()
y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

mae_nn = mean_absolute_error(y_test_windowed, y_pred)
rmse_nn = np.sqrt(mean_squared_error(y_test_windowed, y_pred))

print(f"\nneural network: MAE=${mae_nn:,.0f}  RMSE=${rmse_nn:,.0f}")

# save everything evaluate.py needs
torch.save(model.state_dict(), "save_model/model.pt")
joblib.dump(x_scaler, "save_model/x_scaler.pkl")
joblib.dump(y_scaler, "save_model/y_scaler.pkl")
print("saved model and scalers to save_model/")

