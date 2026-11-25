
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load datasets
uni = np.load("data/processed/lstm_data.npz")
multi = np.load("data/processed/multivariate_lstm_data.npz")

# 2. Load trained models
lstm = tf.keras.models.load_model("models/traffic_lstm.keras")
multi_lstm = tf.keras.models.load_model("models/traffic_multivariate_lstm.keras")
transformer = tf.keras.models.load_model("models/traffic_transformer.keras")

# 3. Load scalers
uni_scaler = joblib.load("models/traffic_scaler.pkl")
multi_scaler = joblib.load("models/multivariate_target_scaler.pkl")

# 4. Generate predictions
pred_lstm = lstm.predict(uni["X_test"], verbose=0)
pred_multi = multi_lstm.predict(multi["X_test"], verbose=0)
pred_transformer = transformer.predict(multi["X_test"], verbose=0)

# 5. Convert predictions to original traffic-volume scale
actual = uni_scaler.inverse_transform(
    uni["y_test"].reshape(-1, 1)
).flatten()

actual_multi = multi_scaler.inverse_transform(
    multi["y_test"].reshape(-1, 1)
).flatten()

if not np.allclose(actual, actual_multi, atol=0.1):
    raise ValueError("The test targets do not match between datasets.")

pred_lstm = uni_scaler.inverse_transform(pred_lstm).flatten()
pred_multi = multi_scaler.inverse_transform(pred_multi).flatten()
pred_transformer = multi_scaler.inverse_transform(pred_transformer).flatten()

# 6. Calculate metrics
predictions = {
    "Original LSTM": pred_lstm,
    "Multivariate LSTM": pred_multi,
    "Transformer": pred_transformer
}

results = []

for name, prediction in predictions.items():
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(actual, prediction),
        "RMSE": np.sqrt(mean_squared_error(actual, prediction)),
        "R2": r2_score(actual, prediction)
    })

comparison = pd.DataFrame(results)

print("\n========== THREE MODEL COMPARISON ==========")
print(comparison.round(4).to_string(index=False))

# 7. Save comparison table
comparison.to_csv(
    "models/three_model_comparison.csv",
    index=False
)

# 8. Plot metric comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ["MAE", "RMSE", "R2"]
titles = ["Mean Absolute Error", "Root Mean Squared Error", "R² Score"]

for ax, metric, title in zip(axes, metrics, titles):
    ax.bar(comparison["Model"], comparison[metric])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("models/three_model_metrics.png", dpi=300)
plt.show()

# 9. Plot actual vs predicted traffic
plt.figure(figsize=(14, 6))

sample_count = 300
plt.plot(actual[:sample_count], label="Actual", linewidth=2)
plt.plot(pred_lstm[:sample_count], label="Original LSTM")
plt.plot(pred_multi[:sample_count], label="Multivariate LSTM")
plt.plot(pred_transformer[:sample_count], label="Transformer")

plt.title("Actual vs Predicted Traffic Volume")
plt.xlabel("Test sample")
plt.ylabel("Traffic volume")
plt.legend()
plt.tight_layout()

plt.savefig("models/three_model_predictions.png", dpi=300)
plt.show()

print("\nAll comparison files saved successfully!")
