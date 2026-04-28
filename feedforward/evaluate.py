import numpy as np
import matplotlib.pyplot as plt


# load the predictions that train.py saved
y_pred = np.load("../results/y_pred.npy")
y_test = np.load("../results/y_test.npy")
test_dates = np.load("../results/test_dates.npy", allow_pickle=True)
baseline_pred = np.load("../results/baseline_pred.npy")


# ---- full test period plot ----

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test, label="Actual Sales", color="steelblue", linewidth=1.5)
plt.plot(test_dates, y_pred, label="NN Predictions", color="tomato", linestyle="--", linewidth=1.5)
plt.plot(test_dates, baseline_pred, label="Baseline (7-day avg)", color="lightgray", linewidth=1)
plt.title("Actual vs Predicted Daily Sales")
plt.xlabel("Date")
plt.ylabel("Sales ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/predictions_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved predictions_vs_actual.png")


# ---- march 2017 zoom ----

# convert to datetime so we can filter
dates_dt = np.array(test_dates, dtype="datetime64[ns]")
march_mask = (dates_dt >= np.datetime64("2017-03-01")) & (dates_dt < np.datetime64("2017-04-01"))

plt.figure(figsize=(12, 5))
plt.plot(test_dates[march_mask], y_test[march_mask], label="Actual Sales",
         color="steelblue", linewidth=2, marker="o", markersize=4)
plt.plot(test_dates[march_mask], y_pred[march_mask], label="NN Predictions",
         color="tomato", linestyle="--", linewidth=2, marker="x", markersize=4)
plt.plot(test_dates[march_mask], baseline_pred[march_mask], label="Baseline (7-day avg)",
         color="lightgray", linewidth=1)
plt.title("March 2017 — Weekly Cycle Close-Up")
plt.xlabel("Date")
plt.ylabel("Sales ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/march_2017_zoom.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved march_2017_zoom.png")


# ---- results summary ----

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nneural network: MAE=${mae:,.0f}  RMSE=${rmse:,.0f}")

# check if model underpredicts more early vs late
residuals = y_test - y_pred
print(f"\nresiduals (positive = model underpredicted):")
print(f"  jan-mar: ${residuals[:90].mean():,.0f}")
print(f"  jun-aug: ${residuals[-75:].mean():,.0f}")
