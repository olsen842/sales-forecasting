import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_prep import load_and_prepare, split_and_scale


def evaluate_baselines(test_df, y_test):
    results = {}

    # baseline 1: just predict yesterday's sales
    pred_lag1 = test_df["lag_1"].values
    mae = mean_absolute_error(y_test, pred_lag1)
    rmse = np.sqrt(mean_squared_error(y_test, pred_lag1))
    results["Yesterday (lag-1)"] = (mae, rmse)

    # baseline 2: 7-day rolling average
    pred_roll7 = test_df["rolling_7"].values
    mae = mean_absolute_error(y_test, pred_roll7)
    rmse = np.sqrt(mean_squared_error(y_test, pred_roll7))
    results["7-day rolling avg"] = (mae, rmse)

    # TODO: could add a 30-day rolling avg too but probably not worth it

    return results


if __name__ == "__main__":
    daily = load_and_prepare()
    data = split_and_scale(daily)

    results = evaluate_baselines(data["test_df"], data["y_test"])

    print("baseline results:")
    for name, (mae, rmse) in results.items():
        print(f"  {name}: MAE=${mae:,.0f}  RMSE=${rmse:,.0f}")
