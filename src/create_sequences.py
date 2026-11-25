
import pandas as pd
import numpy as np
from pathlib import Path

# Load cleaned data
project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "processed" / "traffic_cleaned.csv"

df = pd.read_csv(file_path)
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)

# Define sequence length
sequence_length = 24

# Identify continuous hourly segments
df["new_segment"] = (
    df["date_time"].diff() != pd.Timedelta(hours=1)
)
df["segment_id"] = df["new_segment"].cumsum()

X = []
y = []
target_times = []

# Generate sequences without crossing time gaps
for segment_id, segment in df.groupby("segment_id"):

    traffic = segment["traffic_volume"].to_numpy()
    timestamps = segment["date_time"].to_numpy()

    if len(traffic) <= sequence_length:
        continue

    for i in range(len(traffic) - sequence_length):

        X.append(traffic[i:i + sequence_length])
        y.append(traffic[i + sequence_length])
        target_times.append(timestamps[i + sequence_length])

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)
target_times = np.array(target_times)

# LSTM expects 3 dimensions: samples, time steps, features
X = X.reshape(X.shape[0], X.shape[1], 1)

print("\n========== LSTM SEQUENCE DATA ==========")
print("Input shape:", X.shape)
print("Target shape:", y.shape)
print("Number of sequences:", len(X))
print("Sequence length:", sequence_length)
print("Features per hour:", X.shape[2])

print("\nExample input (previous 24 hours):")
print(X[0].flatten())

print("\nExpected output (next hour):")
print(y[0])

print("\nTarget timestamp:")
print(target_times[0])
