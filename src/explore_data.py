
import pandas as pd
from pathlib import Path

# Locate the dataset
project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "raw" / "traffic.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Display basic information
print("\n========== DATASET OVERVIEW ==========")

print("\n1. Dataset Shape:")
print(df.shape)

print("\n2. Column Names:")
print(df.columns.tolist())

print("\n3. First 5 Records:")
print(df.head())

print("\n4. Dataset Information:")
df.info()

print("\n5. Missing Values:")
print(df.isnull().sum())

print("\n========== DETAILED DATASET ANALYSIS ==========")

print("\n1. Holiday Categories:")
print(df["holiday"].value_counts(dropna=False))

print("\n2. Numerical Statistics:")
print(df[["temp", "rain_1h", "snow_1h",
          "clouds_all", "traffic_volume"]].describe())

print("\n3. Duplicate Records:")
print(df.duplicated().sum())

print("\n4. Date Range:")
print("Start:", df["date_time"].min())
print("End:", df["date_time"].max())

print("\n5. Traffic Volume Distribution:")
print("Minimum:", df["traffic_volume"].min())
print("Maximum:", df["traffic_volume"].max())
print("Average:", df["traffic_volume"].mean())

print("\n========== OUTLIER INVESTIGATION ==========")

# Check temperature values equal to zero
print("\n1. Records with temperature = 0:")
print(df[df["temp"] == 0])

# Check the 10 highest rainfall values
print("\n2. Top 10 rainfall records:")
print(
    df.nlargest(10, "rain_1h")[
        ["date_time", "rain_1h", "weather_main",
         "weather_description", "traffic_volume"]
    ]
)

# Check records with zero traffic
print("\n3. Records with zero traffic:")
print("Count:", (df["traffic_volume"] == 0).sum())

# Check the highest traffic values
print("\n4. Top 10 traffic volume records:")
print(df.nlargest(10, "traffic_volume")[
    ["date_time", "traffic_volume", "weather_main"]
])

print("\n========== FINAL DATA QUALITY CHECK ==========")

# 1. Inspect records around invalid temperature values
print("\n1. Temperature = 0 records:")
zero_temp_indices = df.index[df["temp"] == 0]

for index in zero_temp_indices:
    print(df.loc[max(0, index-1):index+1,
                 ["date_time", "temp", "traffic_volume"]])

# 2. Inspect the extreme rainfall record
print("\n2. Extreme rainfall record:")
print(df[df["rain_1h"] > 100])

# 3. Find timestamps appearing more than once
print("\n3. Repeated timestamps:")
repeated = df[df.duplicated("date_time", keep=False)]

print("Number of records with repeated timestamps:", len(repeated))
print("Number of repeated timestamps:",
      repeated["date_time"].nunique())

print(repeated[
    ["date_time", "weather_main", "weather_description",
     "traffic_volume"]
].head(20))

# 4. Check exact duplicate rows
print("\n4. Exact duplicate count:", df.duplicated().sum())

print("\n========== TIMESTAMP CONSISTENCY CHECK ==========")

# Check whether repeated timestamps have different traffic values
traffic_variation = (
    df.groupby("date_time")["traffic_volume"].nunique()
)

print("\n1. Timestamps with different traffic values:")
print((traffic_variation > 1).sum())

# Check the number of invalid temperature values
print("\n2. Zero temperature records:")
print((df["temp"] == 0).sum())

# Check the number of extreme rainfall records
print("\n3. Rainfall above 100:")
print((df["rain_1h"] > 100).sum())