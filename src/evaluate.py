import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_prep import load_and_prepare, split_and_scale
from baseline import evaluate_baselines
from train_model import build_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_trained_model(n_features, path=None):
    if path is None:
        path = RESULTS_DIR / "model.pt"
    model = build_model(n_features)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def get_predictions(model, X_test_scaled, y_scaler):
    X_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(X_t).numpy()
    return y_scaler.inverse_transform(pred_scaled).flatten()


def plot_full_period(test_dates, y_actual, y_nn, y_baseline, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_dates, y_actual, label="Actual", color="steelblue", linewidth=1.5)
    ax.plot(test_dates, y_nn, label="Neural Network", color="tomato", linestyle="--", linewidth=1.5)
    ax.plot(test_dates, y_baseline, label="7-day avg baseline", color="lightgray", linewidth=1)
    ax.set_title("Actual vs Predicted Daily Sales (Jan-Aug 2017)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {save_path.name}")


def plot_march_zoom(test_df, y_actual, y_nn, y_baseline, save_path):
    # zoom into march to show the weekly cycle
    mask = (test_df["date"] >= "2017-03-01") & (test_df["date"] < "2017-04-01")
    dates = test_df.loc[mask, "date"].values
    actual = y_actual[mask.values]
    pred = y_nn[mask.values]
    bl = y_baseline[mask.values]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, actual, label="Actual", color="steelblue", linewidth=2, marker="o", markersize=4)
    ax.plot(dates, pred, label="Neural Network", color="tomato", linestyle="--", linewidth=2, marker="x", markersize=4)
    ax.plot(dates, bl, label="7-day avg", color="lightgray", linewidth=1)
    ax.set_title("March 2017 - weekly cycle")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {save_path.name}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)

    daily = load_and_prepare()
    data = split_and_scale(daily)

    # baselines
    baselines = evaluate_baselines(data["test_df"], data["y_test"])
    print("baselines:")
    for name, (mae, rmse) in baselines.items():
        print(f"  {name}: MAE=${mae:,.0f}")

    # neural network
    model = load_trained_model(len(data["features"]))
    y_nn = get_predictions(model, data["X_test"], data["y_scaler"])

    nn_mae = mean_absolute_error(data["y_test"], y_nn)
    nn_rmse = np.sqrt(mean_squared_error(data["y_test"], y_nn))
    print(f"\nneural network: MAE=${nn_mae:,.0f}  RMSE=${nn_rmse:,.0f}")

    # plots
    test_dates = data["test_df"]["date"].values
    y_bl = data["test_df"]["rolling_7"].values

    plot_full_period(test_dates, data["y_test"], y_nn, y_bl,
                     RESULTS_DIR / "predictions_vs_actual.png")
    plot_march_zoom(data["test_df"], data["y_test"], y_nn, y_bl,
                    RESULTS_DIR / "march_2017_zoom.png")

    # check if model is still underpredicting more at the start than the end
    residuals = data["y_test"] - y_nn
    print(f"\nresiduals (positive = underpredicting):")
    print(f"  jan-mar avg: ${residuals[:90].mean():,.0f}")
    print(f"  jun-aug avg: ${residuals[-75:].mean():,.0f}")
