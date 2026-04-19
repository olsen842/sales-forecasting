import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"

# these are the features I ended up using after some trial and error
FEATURES = ["lag_1", "lag_7", "rolling_7", "rolling_30", "dayofweek", "month", "is_weekend"]

CUTOFF = pd.Timestamp("2017-01-01")


def load_and_prepare(data_path=DATA_PATH):
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # sum everything into one daily number
    daily = df.groupby("date")["sales"].sum().reset_index()

    # lag features - yesterday and same day last week
    daily["lag_1"] = daily["sales"].shift(1)
    daily["lag_7"] = daily["sales"].shift(7)

    # rolling averages
    daily["rolling_7"] = daily["sales"].rolling(7).mean()
    daily["rolling_30"] = daily["sales"].rolling(30).mean()

    # calendar stuff
    daily["dayofweek"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = (daily["dayofweek"] >= 5).astype(int)

    # drop the first 30 rows that have NaN from rolling_30
    daily = daily.dropna().reset_index(drop=True)
    return daily


def split_and_scale(daily, cutoff=CUTOFF, features=FEATURES):
    train_df = daily[daily["date"] < cutoff]
    test_df = daily[daily["date"] >= cutoff]

    X_train = train_df[features]
    y_train = train_df["sales"]
    X_test = test_df[features]
    y_test = test_df["sales"]

    # scale features
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    # scale target too - without this training didn't converge at all
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "y_train_scaled": y_train_scaled,
        "y_test_scaled": y_test_scaled,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "train_df": train_df,
        "test_df": test_df,
        "features": features,
    }


if __name__ == "__main__":
    daily = load_and_prepare()
    print(f"loaded {len(daily)} days of data")
    print(f"date range: {daily['date'].min().date()} to {daily['date'].max().date()}")

    data = split_and_scale(daily)
    print(f"train: {len(data['train_df'])} days")
    print(f"test:  {len(data['test_df'])} days")
