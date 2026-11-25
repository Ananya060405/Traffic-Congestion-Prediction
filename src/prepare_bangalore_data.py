"""Prepare Bengaluru road-level observations for next-day prediction.

The source CSV stays untouched. Only consecutive-day pairs are retained because
the target is explicitly the following calendar day.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "bangalore" / "archive (3)" / "Banglore_traffic_Dataset.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_FILE = OUT_DIR / "bangalore_next_day_training.csv"

REQUIRED = [
    "Date", "Area Name", "Road/Intersection Name", "Traffic Volume",
    "Average Speed", "Travel Time Index", "Congestion Level",
    "Road Capacity Utilization", "Incident Reports", "Environmental Impact",
    "Public Transport Usage", "Traffic Signal Compliance", "Parking Usage",
    "Pedestrian and Cyclist Count", "Weather Conditions",
    "Roadwork and Construction Activity",
]

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {SOURCE}\n"
            "Keep the downloaded CSV at data/raw/bangalore/archive (3)/Banglore_traffic_Dataset.csv"
        )

    df = pd.read_csv(SOURCE)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Some Date values could not be parsed.")

    df = df.sort_values(["Road/Intersection Name", "Date"]).reset_index(drop=True)
    grouped = df.groupby("Road/Intersection Name", sort=False)
    df["Target Date"] = grouped["Date"].shift(-1)
    df["Gap Days"] = (df["Target Date"] - df["Date"]).dt.days
    df["Target Congestion Level"] = grouped["Congestion Level"].shift(-1)
    df["Target Traffic Volume"] = grouped["Traffic Volume"].shift(-1)

    # The target is the next calendar day, not merely the next available record.
    pairs = df.loc[df["Gap Days"] == 1].copy()
    if pairs.empty:
        raise ValueError("No consecutive-day road observations were found.")

    pairs["Target Year"] = pairs["Target Date"].dt.year
    pairs["Target Month"] = pairs["Target Date"].dt.month
    pairs["Target Day"] = pairs["Target Date"].dt.day
    pairs["Target DayOfWeek"] = pairs["Target Date"].dt.dayofweek
    pairs["Target IsWeekend"] = (pairs["Target DayOfWeek"] >= 5).astype(int)

    # Explicitly name source-day features so the training script cannot
    # accidentally use next-day target values as inputs.
    source_features = [
        "Area Name", "Road/Intersection Name",
        "Traffic Volume", "Average Speed", "Travel Time Index",
        "Congestion Level", "Road Capacity Utilization", "Incident Reports",
        "Environmental Impact", "Public Transport Usage",
        "Traffic Signal Compliance", "Parking Usage",
        "Pedestrian and Cyclist Count", "Weather Conditions",
        "Roadwork and Construction Activity",
    ]
    rename = {c: f"Previous Day - {c}" for c in source_features}
    pairs = pairs.rename(columns=rename)

    output_columns = (
        ["Date", "Target Date", "Gap Days"]
        + [rename[c] for c in source_features]
        + ["Target Year", "Target Month", "Target Day",
           "Target DayOfWeek", "Target IsWeekend",
           "Target Congestion Level", "Target Traffic Volume"]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs[output_columns].to_csv(OUT_FILE, index=False)

    print(f"Saved: {OUT_FILE}")
    print(f"Consecutive-day training examples: {len(pairs):,}")
    print(f"Roads represented: {pairs['Previous Day - Road/Intersection Name'].nunique()}")
    print(f"Target date range: {pairs['Target Date'].min().date()} to {pairs['Target Date'].max().date()}")
    print("\nExamples per road:")
    print(pairs.groupby("Previous Day - Road/Intersection Name").size().to_string())

if __name__ == "__main__":
    main()
