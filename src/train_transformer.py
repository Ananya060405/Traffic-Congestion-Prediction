
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# 1. Load prepared data
data = np.load(
    "data/processed/multivariate_lstm_data.npz",
    allow_pickle=True
)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Testing shape:", X_test.shape)

# 2. Load target scaler
target_scaler = joblib.load(
    "models/multivariate_target_scaler.pkl"
)

# 3. Positional encoding
def positional_encoding(length, depth):
    positions = np.arange(length)[:, np.newaxis]
    dimensions = np.arange(depth)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (dimensions // 2)) / np.float32(depth)
    )

    angle_rads = positions * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.cast(angle_rads, dtype=tf.float32)


# 4. Transformer encoder block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):

    attention = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)

    attention = layers.Dropout(dropout)(attention)

    x = layers.LayerNormalization(epsilon=1e-6)(
        inputs + attention
    )

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(inputs.shape[-1])(ff)

    return layers.LayerNormalization(epsilon=1e-6)(
        x + ff
    )


# 5. Build Transformer model
def build_transformer():

    sequence_length = X_train.shape[1]
    number_of_features = X_train.shape[2]

    inputs = layers.Input(
        shape=(sequence_length, number_of_features)
    )

    x = layers.Dense(64)(inputs)

    position = positional_encoding(sequence_length, 64)
    x = x + position

    x = transformer_encoder(
        x,
        head_size=32,
        num_heads=4,
        ff_dim=128,
        dropout=0.1
    )

    x = transformer_encoder(
        x,
        head_size=32,
        num_heads=4,
        ff_dim=128,
        dropout=0.1
    )

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model


# 6. Create and display model
model = build_transformer()

model.summary()

# 7. Train model
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=0.00001
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 8. Predict test data
predictions = model.predict(X_test)

# 9. Convert scaled values back to traffic volume
actual = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

predicted = target_scaler.inverse_transform(
    predictions
).flatten()

# 10. Calculate evaluation metrics
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("\n========== TRANSFORMER RESULTS ==========")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")

# 11. Save model
os.makedirs("models", exist_ok=True)

model.save("models/traffic_transformer.keras")

# 12. Save predictions and metrics
pd.DataFrame({
    "Actual": actual,
    "Predicted": predicted
}).to_csv(
    "models/transformer_predictions.csv",
    index=False
)

pd.DataFrame([{
    "Model": "Transformer",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}]).to_csv(
    "models/transformer_metrics.csv",
    index=False
)

print("\nTransformer model and results saved successfully!")
