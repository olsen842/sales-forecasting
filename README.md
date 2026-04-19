# Grocery Sales Forecasting with PyTorch

**A neural network that predicts daily grocery sales 43% more accurately than simple baselines, built end-to-end with PyTorch.**

![Actual vs Predicted Daily Sales](results/predictions_vs_actual.png)

## Problem

Grocery stores need to know how much they'll sell tomorrow. Over-order and food spoils. Under-order and shelves are empty. This project uses a neural network to predict total daily sales for a large grocery chain.

## Dataset

[**Corporación Favorita — Store Sales**](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) from Kaggle. About 4.5 years of daily sales (Jan 2013 – Aug 2017) across 54 stores and 33 product families — roughly 3 million rows. I aggregated everything into a single daily total to keep it manageable as one time series (1,684 days).

## Approach

1. **Aggregation** — summed sales across all stores and product families to get one number per day
2. **Feature engineering** — created lag features (1-day, 7-day), rolling averages (7-day, 30-day), and calendar features (day of week, month, weekend flag)
3. **Baselines** — tested two naive models first: "tomorrow = today's sales" and "tomorrow = average of last 7 days"
4. **Neural network** — a feedforward network (32 → 16 → 1, ReLU activations, no output activation) trained with Adam optimizer on MSE loss for 350 epochs

## Results

| Model | MAE ($) | RMSE ($) | vs Best Baseline |
|-------|---------|----------|------------------|
| Yesterday's sales (lag-1) | 162,294 | 221,797 | — |
| 7-day rolling average | 137,617 | 174,210 | — |
| **Neural Network** | **78,780** | **126,666** | **−43% MAE** |

## Visualizations

### Full test period (Jan–Aug 2017)
![Full period predictions](results/predictions_vs_actual.png)

### March 2017 — weekly cycle close-up
![March 2017 zoom](results/march_2017_zoom.png)

The March close-up shows the neural network tracking the weekly sales pattern (Sunday peaks, midweek dips) while the rolling average baseline smooths everything flat.

## What I Learned

- **Weekend sales are ~39% higher than weekdays** — I verified this numerically. The `dayofweek` and `is_weekend` features give the neural network its biggest advantage over the rolling average baseline, which can't capture weekly cycles
- **Lag features solve the trend problem** — the training data (2013–2016) has much lower sales than the test period (2017). I expected the model to systematically underpredict, but `lag_1` and `lag_7` anchor predictions to recent values, so it adapts. Average residual was +$30,825 in Jan–Mar but only +$4,628 in Jun–Aug
- **Scaling the target was critical** — my first training attempt produced $873K MAE (worse than predicting zero) because the raw dollar values were too large for the network to learn from. Scaling y with StandardScaler fixed it immediately
- **The model misses holidays badly** — New Year's Day sales drop to nearly $0 (stores are closed), but the model predicts ~$700K because it has no holiday feature. Same for other Ecuadorian holidays that cause big spikes or dips
- **No sigmoid on the output** — this is regression, not classification. Unlike my previous project (a binary classifier), the output layer has no activation function so it can predict any dollar amount

## What I'd Do Next

- **Add holiday features** — the Kaggle dataset includes `holidays_events.csv`. Adding an `is_holiday` flag would fix the model's worst predictions (like New Year's Day)
- **Add oil prices** — Ecuador's economy is oil-dependent, so oil price data (also in the dataset) could explain some of the sales variance
- **Per-store or per-family models** — right now I aggregate everything into one number. Building separate models per product family would be more useful in practice
- **Try an LSTM** — a recurrent network could learn temporal patterns directly from sequences instead of relying on hand-crafted lag features

## How to Run

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd sales-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data
#    Go to https://www.kaggle.com/competitions/store-sales-time-series-forecasting
#    Download train.csv and place it in data/train.csv

# 4. Run the pipeline
cd src
python train.py           # loads data, builds features, trains model, saves results
python evaluate.py        # generates plots and prints final metrics
```

## Project Structure

```
sales-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                    (gitignored)
│   └── train.csv
├── fills/
│   ├── train.py             data loading, features, baselines, NN training
│   └── evaluate.py          plots + final results
└── results/
    ├── predictions_vs_actual.png
    └── march_2017_zoom.png
```
