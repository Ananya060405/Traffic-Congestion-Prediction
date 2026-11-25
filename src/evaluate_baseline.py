
import numpy as np
import joblib
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_root = Path(__file__).resolve().parent.parent

# Load prepared test data
data = np.load(
    project_root / "data" / "processed" / "lstm_data.npz"
)

X_test = data["X_test"]
y_test = data["y_test"]

# Load trained model and scaler
model = tf.keras.models.load_model(
    project_root / "models" / "traffic_lstm.keras"
)

scaler = joblib.load(
    project_root / "models" / "traffic_scaler.pkl"
)

# LSTM predictions (scaled values)
lstm_scaled = model.predict(X_test, verbose=0)

# Convert predictions and actual values to original traffic units
lstm_predictions = scaler.inverse_transform(
    lstm_scaled
).flatten()

actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

# Baseline: predict next hour using the most recent hour
baseline_scaled = X_test[:, -1, 0].reshape(-1, 1)

baseline_predictions = scaler.inverse_transform(
    baseline_scaled
).flatten()

# Evaluation function
def evaluate(name, predictions):
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)

    print(f"\n{name}")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

print("\n========== MODEL COMPARISON ==========")

evaluate("LSTM MODEL", lstm_predictions)
evaluate("PERSISTENCE BASELINE", baseline_predictions)
