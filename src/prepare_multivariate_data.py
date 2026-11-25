
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# 1. Load cleaned dataset
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "processed" / "traffic_cleaned.csv"
output_folder = project_root / "data" / "processed"

df = pd.read_csv(input_file)
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)

# 2. Create holiday feature
df["is_holiday"] = (df["holiday"] != "No Holiday").astype(int)

# 3. Create cyclical time features
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

# 4. One-hot encode weather categories
weather_encoded = pd.get_dummies(
    df["weather_main"],
    prefix="weather",
    dtype=int
)

df = pd.concat([df, weather_encoded], axis=1)

# 5. Select model input features
feature_columns = [
    "traffic_volume",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "is_holiday",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos"
] + list(weather_encoded.columns)

target_column = "traffic_volume"

print("\nNumber of input features:", len(feature_columns))
print("Input features:", feature_columns)

# 6. Identify continuous hourly segments
df["new_segment"] = (
    df["date_time"].diff() != pd.Timedelta(hours=1)
)
df["segment_id"] = df["new_segment"].cumsum()

# 7. Generate 24-hour sequences
sequence_length = 24

X = []
y = []
target_times = []

for _, segment in df.groupby("segment_id"):

    features = segment[feature_columns].to_numpy(dtype=float)
    targets = segment[target_column].to_numpy(dtype=float)
    timestamps = segment["date_time"].to_numpy()

    for i in range(len(segment) - sequence_length):

        X.append(features[i:i + sequence_length])
        y.append(targets[i + sequence_length])
        target_times.append(timestamps[i + sequence_length])

X = np.array(X)
y = np.array(y)
target_times = np.array(target_times)

# 8. Sort sequences chronologically
order = np.argsort(target_times)

X = X[order]
y = y[order]
target_times = target_times[order]

# 9. Chronological split: 70%, 15%, 15%
total = len(X)

train_end = int(total * 0.70)
val_end = int(total * 0.85)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

# 10. Fit scalers using training-period records only
first_val_time = pd.Timestamp(target_times[train_end])

training_rows = df.loc[
    df["date_time"] < first_val_time
]

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

feature_scaler.fit(training_rows[feature_columns])
target_scaler.fit(training_rows[[target_column]])

# 11. Scale input features
def scale_inputs(data):
    original_shape = data.shape

    flat_data = pd.DataFrame(
        data.reshape(-1, len(feature_columns)),
        columns=feature_columns
    )

    scaled = feature_scaler.transform(flat_data)

    return scaled.reshape(original_shape)

X_train = scale_inputs(X_train)
X_val = scale_inputs(X_val)
X_test = scale_inputs(X_test)

# 12. Scale target values
y_train = target_scaler.transform(
    y_train.reshape(-1, 1)
).flatten()

y_val = target_scaler.transform(
    y_val.reshape(-1, 1)
).flatten()

y_test = target_scaler.transform(
    y_test.reshape(-1, 1)
).flatten()

# 13. Save prepared datasets
np.savez(
    output_folder / "multivariate_lstm_data.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    target_times=target_times,
    feature_columns=np.array(feature_columns)
)

joblib.dump(
    feature_scaler,
    project_root / "models" / "multivariate_feature_scaler.pkl"
)

joblib.dump(
    target_scaler,
    project_root / "models" / "multivariate_target_scaler.pkl"
)

# 14. Display results
print("\n========== MULTIVARIATE DATA PREPARATION ==========")

print("Total sequences:", total)
print("Number of features:", len(feature_columns))

print("\nTraining:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nValidation:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("\nTesting:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining target dates:")
print(target_times[0], "to", target_times[train_end - 1])

print("\nValidation target dates:")
print(target_times[train_end], "to", target_times[val_end - 1])

print("\nTesting target dates:")
print(target_times[val_end], "to", target_times[-1])

print("\nSaved multivariate dataset successfully!")
