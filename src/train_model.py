import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_prep import load_and_prepare, split_and_scale

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# hyperparameters - tried a few combos, these worked best
EPOCHS = 350
LR = 0.001
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_model(n_features):
    # 3 layers felt like enough, adding more didn't really help
    model = nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        # no sigmoid here - this is regression not classification
    )
    return model


def train(model, X_train, y_train, X_test, y_test, epochs=EPOCHS, lr=LR):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_test), y_test).item()
            print(f"epoch {epoch+1}/{epochs}  train loss: {loss.item():.4f}  test loss: {val_loss:.4f}")

    return model


if __name__ == "__main__":
    daily = load_and_prepare()
    data = split_and_scale(daily)

    X_train_t = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train_t = torch.tensor(data["y_train_scaled"], dtype=torch.float32)
    X_test_t = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test_t = torch.tensor(data["y_test_scaled"], dtype=torch.float32)

    n_features = len(data["features"])
    model = build_model(n_features)
    print(f"training with {n_features} features for {EPOCHS} epochs...")

    model = train(model, X_train_t, y_train_t, X_test_t, y_test_t)

    # get predictions back in original dollar scale
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).numpy()
    y_pred = data["y_scaler"].inverse_transform(y_pred_scaled).flatten()

    mae = mean_absolute_error(data["y_test"], y_pred)
    rmse = np.sqrt(mean_squared_error(data["y_test"], y_pred))
    print(f"\nresults on test set:")
    print(f"  MAE  = ${mae:,.0f}")
    print(f"  RMSE = ${rmse:,.0f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), RESULTS_DIR / "model.pt")
    print(f"model saved")
