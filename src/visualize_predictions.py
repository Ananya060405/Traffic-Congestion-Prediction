
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

# Load test data
data = np.load(
    project_root / "data" / "processed" / "lstm_data.npz"
)

X_test = data["X_test"]
y_test = data["y_test"]

# Load model and scaler
model = tf.keras.models.load_model(
    project_root / "models" / "traffic_lstm.keras"
)

scaler = joblib.load(
    project_root / "models" / "traffic_scaler.pkl"
)

# Generate predictions
predicted_scaled = model.predict(X_test, verbose=0)

# Convert values back to actual traffic volume
predicted = scaler.inverse_transform(
    predicted_scaled
).flatten()

actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

# Plot first 300 test samples in chronological sample order
sample_count = min(300, len(actual))

plt.figure(figsize=(14, 6))

plt.plot(
    actual[:sample_count],
    label="Actual Traffic",
    linewidth=2
)

plt.plot(
    predicted[:sample_count],
    label="Predicted Traffic",
    linewidth=2
)

plt.title("LSTM: Actual vs Predicted Traffic Volume")
plt.xlabel("Test Sample (chronological order)")
plt.ylabel("Traffic Volume (vehicles/hour)")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save graph
output_folder = project_root / "models"
output_folder.mkdir(exist_ok=True)

plt.savefig(
    output_folder / "lstm_predictions.png",
    dpi=300
)

plt.show()

print("Graph saved at:", output_folder / "lstm_predictions.png")
