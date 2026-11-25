
import numpy as np
import tensorflow as tf
import joblib

from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Locate project files
project_root = Path(__file__).resolve().parent.parent

data_file = project_root / "data" / "processed" / "lstm_data.npz"
scaler_file = project_root / "models" / "traffic_scaler.pkl"
model_file = project_root / "models" / "traffic_lstm.keras"

# 2. Load prepared data
print("Loading prepared LSTM data...")

data = np.load(data_file)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Testing data:", X_test.shape)

# 3. Build LSTM model
model = Sequential([
    Input(shape=(24, 1)),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

# 4. Compile model
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

print("\n========== MODEL SUMMARY ==========")
model.summary()

# 5. Configure callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

# 6. Train model
print("\n========== STARTING LSTM TRAINING ==========")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 7. Save trained model
model.save(model_file)

print("\nModel saved at:", model_file)

# 8. Evaluate on unseen test data
print("\n========== TESTING MODEL ==========")

test_loss, test_mae_scaled = model.evaluate(
    X_test, y_test, verbose=0
)

# Convert predictions back to original traffic-volume scale
scaler = joblib.load(scaler_file)

predictions_scaled = model.predict(X_test, verbose=0)

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

predictions_actual = scaler.inverse_transform(
    predictions_scaled
).flatten()

# Calculate evaluation metrics
mae = mean_absolute_error(
    y_test_actual, predictions_actual
)

rmse = np.sqrt(
    mean_squared_error(y_test_actual, predictions_actual)
)

r2 = r2_score(y_test_actual, predictions_actual)

print("\nTest Loss (scaled):", round(test_loss, 4))
print("Test MAE:", round(mae, 2))
print("Test RMSE:", round(rmse, 2))
print("Test R2 Score:", round(r2, 4))

# 9. Show sample predictions
print("\n========== SAMPLE PREDICTIONS ==========")

for i in range(10):
    print(
        f"Actual: {y_test_actual[i]:.0f} | "
        f"Predicted: {predictions_actual[i]:.0f}"
    )

print("\nLSTM training and evaluation complete!")
