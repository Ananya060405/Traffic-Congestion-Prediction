
import pandas as pd
from pathlib import Path

# Locate and load cleaned dataset
project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "processed" / "traffic_cleaned.csv"

df = pd.read_csv(file_path)
df["date_time"] = pd.to_datetime(df["date_time"])

# Sort records chronologically
df = df.sort_values("date_time").reset_index(drop=True)

# Calculate time difference between consecutive records
df["time_gap"] = df["date_time"].diff()

# Identify gaps greater than one hour
gaps = df[df["time_gap"] > pd.Timedelta(hours=1)]

print("\n========== TIME GAP ANALYSIS ==========")

print("\nTotal records:", len(df))
print("Consecutive one-hour intervals:",
      (df["time_gap"] == pd.Timedelta(hours=1)).sum())

print("Gaps greater than one hour:", len(gaps))

print("\nLargest time gap:")
print(df["time_gap"].max())

print("\nMost common time gaps:")
print(df["time_gap"].value_counts().head(10))

print("\nFirst 10 records with gaps:")
print(gaps[["date_time", "time_gap"]].head(10))
