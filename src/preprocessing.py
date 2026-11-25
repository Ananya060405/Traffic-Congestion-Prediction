
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Locate project folders
project_root = Path(__file__).resolve().parent.parent

input_file = project_root / "data" / "raw" / "traffic.csv"
output_folder = project_root / "data" / "processed"
output_file = output_folder / "traffic_cleaned.csv"

output_folder.mkdir(parents=True, exist_ok=True)

# 2. Load dataset
print("Loading dataset...")
df = pd.read_csv(input_file)

print("Original shape:", df.shape)

# 3. Remove exact duplicate records
df = df.drop_duplicates()

# 4. Handle missing holiday values
df["holiday"] = df["holiday"].fillna("No Holiday")

# 5. Convert invalid values into missing values
df.loc[df["temp"] == 0, "temp"] = np.nan
df.loc[df["rain_1h"] > 100, "rain_1h"] = np.nan

# 6. Fill invalid numerical values
df["temp"] = df["temp"].interpolate(method="linear")
df["rain_1h"] = df["rain_1h"].fillna(
    df["rain_1h"].median()
)

# 7. Convert date_time to datetime
df["date_time"] = pd.to_datetime(df["date_time"])

# 8. Consolidate repeated timestamps
df = df.groupby("date_time", as_index=False).agg({
    "holiday": "first",
    "temp": "mean",
    "rain_1h": "mean",
    "snow_1h": "mean",
    "clouds_all": "mean",
    "weather_main": "first",
    "weather_description": "first",
    "traffic_volume": "first"
})

# 9. Extract time-based features
df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.day
df["month"] = df["date_time"].dt.month
df["day_of_week"] = df["date_time"].dt.day_of_week
df["year"] = df["date_time"].dt.year
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# 10. Sort chronologically
df = df.sort_values("date_time").reset_index(drop=True)

# 11. Save cleaned dataset
df.to_csv(output_file, index=False)

# 12. Display results
print("\n========== PREPROCESSING COMPLETE ==========")

print("Final shape:", df.shape)
print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nDuplicate timestamps:", df["date_time"].duplicated().sum())

print("\nSaved cleaned dataset at:")
print(output_file)

print("\nFirst 5 cleaned records:")
print(df.head())
