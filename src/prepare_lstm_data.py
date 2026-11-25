
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# 1. Load cleaned dataset
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "processed" / "traffic_cleaned.csv"

output_folder = project_root / "data" / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)

# 2. Create continuous hourly segments
df["new_segment"] = (
    df["date_time"].diff() != pd.Timedelta(hours=1)
)
df["segment_id"] = df["new_segment"].cumsum()

# 3. Generate 24-hour input sequences
sequence_length = 24

X = []
y = []
target_times = []

for _, segment in df.groupby("segment_id"):

    traffic = segment["traffic_volume"].to_numpy()
    timestamps = segment["date_time"].to_numpy()

    for i in range(len(traffic) - sequence_length):
        X.append(traffic[i:i + sequence_length])
        y.append(traffic[i + sequence_length])
        target_times.append(timestamps[i + sequence_length])

X = np.array(X)
y = np.array(y)
target_times = np.array(target_times)

# 4. Sort sequences chronologically
order = np.argsort(target_times)

X = X[order]
y = y[order]
target_times = target_times[order]

# 5. Split chronologically: 70%, 15%, 15%
total = len(X)

train_end = int(total * 0.70)
val_end = int(total * 0.85)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

# 6. Fit scaler only on training-period records
first_val_time = pd.Timestamp(target_times[train_end])

training_values = df.loc[
    df["date_time"] < first_val_time,
    ["traffic_volume"]
].to_numpy()

scaler = MinMaxScaler()

scaler.fit(training_values)

# 7. Scale inputs and targets
X_train = scaler.transform(
    X_train.reshape(-1, 1)
).reshape(X_train.shape)

X_val = scaler.transform(
    X_val.reshape(-1, 1)
).reshape(X_val.shape)

X_test = scaler.transform(
    X_test.reshape(-1, 1)
).reshape(X_test.shape)

y_train = scaler.transform(y_train.reshape(-1, 1)).flatten()
y_val = scaler.transform(y_val.reshape(-1, 1)).flatten()
y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()

# 8. Reshape inputs for LSTM
X_train = X_train.reshape(X_train.shape[0], 24, 1)
X_val = X_val.reshape(X_val.shape[0], 24, 1)
X_test = X_test.reshape(X_test.shape[0], 24, 1)

# 9. Save prepared datasets
np.savez(
    output_folder / "lstm_data.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)

joblib.dump(
    scaler,
    project_root / "models" / "traffic_scaler.pkl"
)

# 10. Display results
print("\n========== LSTM DATA PREPARATION ==========")

print("Total sequences:", total)

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
print(pd.Timestamp(target_times[0]), "to",
      pd.Timestamp(target_times[train_end - 1]))

print("\nValidation target dates:")
print(pd.Timestamp(target_times[train_end]), "to",
      pd.Timestamp(target_times[val_end - 1]))

print("\nTesting target dates:")
print(pd.Timestamp(target_times[val_end]), "to",
      pd.Timestamp(target_times[-1]))

print("\nSaved dataset:", output_folder / "lstm_data.npz")
print("Saved scaler:", project_root / "models" / "traffic_scaler.pkl")
