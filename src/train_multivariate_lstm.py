
import numpy as np
import joblib
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Locate project files
project_root = Path(__file__).resolve().parent.parent
data_file = project_root / "data" / "processed" / "multivariate_lstm_data.npz"

# 2. Load prepared data
print("Loading multivariate LSTM data...")

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

# 3. Build the LSTM architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(24, 24)),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(1)
])

# 4. Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

print("\n========== MODEL SUMMARY ==========")
model.summary()

# 5. Configure callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

# 6. Train model
print("\n========== STARTING MULTIVARIATE LSTM TRAINING ==========")

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
model_path = project_root / "models" / "traffic_multivariate_lstm.keras"

model.save(model_path)

print("\nModel saved at:", model_path)

# 8. Load target scaler
target_scaler = joblib.load(
    project_root / "models" / "multivariate_target_scaler.pkl"
)

# 9. Generate predictions
predictions_scaled = model.predict(X_test, verbose=0)

# Convert predictions and actual values to original traffic units
predictions = target_scaler.inverse_transform(
    predictions_scaled
).flatten()

actual = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

# 10. Calculate evaluation metrics
mae = mean_absolute_error(actual, predictions)

rmse = np.sqrt(
    mean_squared_error(actual, predictions)
)

r2 = r2_score(actual, predictions)

print("\n========== MULTIVARIATE LSTM RESULTS ==========")

print("Test MAE:", round(mae, 2))
print("Test RMSE:", round(rmse, 2))
print("Test R2 Score:", round(r2, 4))

# 11. Display sample predictions
print("\n========== SAMPLE PREDICTIONS ==========")

for i in range(10):
    print(
        f"Actual: {actual[i]:.0f} | "
        f"Predicted: {predictions[i]:.0f}"
    )

print("\nMultivariate LSTM training and evaluation complete!")
