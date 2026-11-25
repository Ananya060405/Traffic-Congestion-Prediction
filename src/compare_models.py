
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Project paths
project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "data" / "processed"
models_folder = project_root / "models"

# 2. Load test datasets
original_data = np.load(data_folder / "lstm_data.npz")
multi_data = np.load(data_folder / "multivariate_lstm_data.npz")

# 3. Load trained models
print("Loading both trained models...")

original_model = tf.keras.models.load_model(
    models_folder / "traffic_lstm.keras"
)

multi_model = tf.keras.models.load_model(
    models_folder / "traffic_multivariate_lstm.keras"
)

# 4. Load scalers
original_scaler = joblib.load(
    models_folder / "traffic_scaler.pkl"
)

multi_target_scaler = joblib.load(
    models_folder / "multivariate_target_scaler.pkl"
)

# 5. Generate predictions
original_predictions_scaled = original_model.predict(
    original_data["X_test"], verbose=0
)

multi_predictions_scaled = multi_model.predict(
    multi_data["X_test"], verbose=0
)

# 6. Convert predictions into actual traffic units
original_predictions = original_scaler.inverse_transform(
    original_predictions_scaled
).flatten()

multi_predictions = multi_target_scaler.inverse_transform(
    multi_predictions_scaled
).flatten()

actual = original_scaler.inverse_transform(
    original_data["y_test"].reshape(-1, 1)
).flatten()

# 7. Calculate metrics
def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }

original_metrics = calculate_metrics(
    actual, original_predictions
)

multi_metrics = calculate_metrics(
    actual, multi_predictions
)

# 8. Create comparison table
comparison = pd.DataFrame(
    [original_metrics, multi_metrics],
    index=["Original LSTM", "Multivariate LSTM"]
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.round(4))

# 9. Save metrics as CSV
comparison.to_csv(
    models_folder / "model_comparison.csv"
)

# 10. Plot metric comparisons
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ["MAE", "RMSE", "R2"]

for ax, metric in zip(axes, metrics):
    comparison[metric].plot(
        kind="bar",
        ax=ax,
        title=metric,
        rot=0
    )
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Original vs Multivariate LSTM")
plt.tight_layout()

plt.savefig(
    models_folder / "model_comparison.png",
    dpi=300
)

plt.show()

# 11. Plot actual vs predicted traffic
sample_count = min(300, len(actual))

plt.figure(figsize=(14, 6))

plt.plot(
    actual[:sample_count],
    label="Actual Traffic"
)

plt.plot(
    original_predictions[:sample_count],
    label="Original LSTM"
)

plt.plot(
    multi_predictions[:sample_count],
    label="Multivariate LSTM"
)

plt.title("Actual vs Predicted Traffic Volume")
plt.xlabel("Test Sample (chronological order)")
plt.ylabel("Traffic Volume (vehicles/hour)")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    models_folder / "all_model_predictions.png",
    dpi=300
)

plt.show()

# 12. Save individual predictions
prediction_results = pd.DataFrame({
    "actual": actual,
    "original_lstm": original_predictions,
    "multivariate_lstm": multi_predictions
})

prediction_results.to_csv(
    models_folder / "model_predictions.csv",
    index=False
)

print("\n========== REPORT SAVED ==========")
print("model_comparison.csv")
print("model_comparison.png")
print("all_model_predictions.png")
print("model_predictions.csv")
