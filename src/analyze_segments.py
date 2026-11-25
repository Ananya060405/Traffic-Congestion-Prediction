
import pandas as pd
from pathlib import Path

# Load cleaned dataset
project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "processed" / "traffic_cleaned.csv"

df = pd.read_csv(file_path)
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)

# Start a new segment whenever the gap is not exactly one hour
df["new_segment"] = (
    df["date_time"].diff() != pd.Timedelta(hours=1)
)

df["segment_id"] = df["new_segment"].cumsum()

# Calculate the size of each continuous segment
segments = df.groupby("segment_id").agg(
    start_time=("date_time", "min"),
    end_time=("date_time", "max"),
    records=("date_time", "count")
)

# Display segment statistics
print("\n========== CONTINUOUS SEGMENT ANALYSIS ==========")

print("Total continuous segments:", len(segments))
print("Longest segment:", segments["records"].max())
print("Shortest segment:", segments["records"].min())

print("\nSegment length statistics:")
print(segments["records"].describe())

# A 24-hour input window + 1 target requires 25 records
usable = segments[segments["records"] >= 25]

print("\nSegments with at least 25 records:", len(usable))
print("Records inside these segments:", usable["records"].sum())

print("\n10 longest continuous segments:")
print(
    segments.sort_values("records", ascending=False).head(10)
)
