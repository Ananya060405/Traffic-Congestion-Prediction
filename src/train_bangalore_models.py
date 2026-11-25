"""Train and evaluate Bengaluru next-day traffic prediction models.

Models use previous-day observations and known calendar features to predict
next-day congestion and traffic volume. The split is chronological.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "bangalore_next_day_training.csv"
MODEL_DIR = ROOT / "models"
METRICS_FILE = ROOT / "reports" / "bangalore_model_metrics.json"

TARGETS = {
    "congestion": "Target Congestion Level",
    "traffic_volume": "Target Traffic Volume",
}
CATEGORICAL = [
    "Previous Day - Area Name",
    "Previous Day - Road/Intersection Name",
    "Previous Day - Weather Conditions",
    "Previous Day - Roadwork and Construction Activity",
]
NUMERIC = [
    "Previous Day - Traffic Volume",
    "Previous Day - Average Speed",
    "Previous Day - Travel Time Index",
    "Previous Day - Congestion Level",
    "Previous Day - Road Capacity Utilization",
    "Previous Day - Incident Reports",
    "Previous Day - Environmental Impact",
    "Previous Day - Public Transport Usage",
    "Previous Day - Traffic Signal Compliance",
    "Previous Day - Parking Usage",
    "Previous Day - Pedestrian and Cyclist Count",
    "Target Year", "Target Month", "Target Day",
    "Target DayOfWeek", "Target IsWeekend",
]
FEATURES = CATEGORICAL + NUMERIC

def make_preprocessor():
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])

def model_candidates():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=350, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=350, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.05, max_depth=2,
            loss="huber", random_state=42
        ),
    }

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {DATA_FILE}\n"
            "Run: python src/prepare_bangalore_data.py"
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["Date", "Target Date"])
    df = df.sort_values("Target Date").reset_index(drop=True)
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} training examples found; refusing to train.")

    # Hold out the latest 20% of target dates. No random shuffling.
    split_at = int(len(df) * 0.8)
    train_df = df.iloc[:split_at].copy()
    test_df = df.iloc[split_at:].copy()
    if train_df["Target Date"].max() >= test_df["Target Date"].min():
        raise ValueError("Chronological split is not strictly separated by date.")

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_metrics = {
        "dataset": str(DATA_FILE.relative_to(ROOT)),
        "training_examples": int(len(train_df)),
        "testing_examples": int(len(test_df)),
        "train_target_date_range": [
            str(train_df["Target Date"].min().date()),
            str(train_df["Target Date"].max().date()),
        ],
        "test_target_date_range": [
            str(test_df["Target Date"].min().date()),
            str(test_df["Target Date"].max().date()),
        ],
        "prediction_scope": "next calendar day, using previous calendar day's observed features",
        "targets": {},
    }

    for target_key, target_column in TARGETS.items():
        y_train = train_df[target_column].astype(float)
        y_test = test_df[target_column].astype(float)
        results = []

        for model_name, estimator in model_candidates().items():
            pipeline = Pipeline([
                ("preprocessor", make_preprocessor()),
                ("model", estimator),
            ])
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            metrics = {
                "model": model_name,
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": rmse,
                "r2": float(r2_score(y_test, predictions)),
            }
            results.append((metrics, pipeline))
            print(
                f"{target_key:14s} | {model_name:18s} | "
                f"MAE {metrics['mae']:.3f} | RMSE {rmse:.3f} | R2 {metrics['r2']:.4f}"
            )

        # Select using lowest MAE on the held-out chronological test set.
        # This is a first baseline; later we can add a separate validation period.
        results.sort(key=lambda item: item[0]["mae"])
        best_metrics, best_pipeline = results[0]
        model_path = MODEL_DIR / f"bangalore_{target_key}_model.joblib"
        joblib.dump(best_pipeline, model_path)

        all_metrics["targets"][target_key] = {
            "best_model": best_metrics["model"],
            "best_model_path": str(model_path.relative_to(ROOT)),
            "model_comparison": [item[0] for item in results],
        }

        # Road-wise diagnostics for the selected model.
        road_results = test_df[["Previous Day - Road/Intersection Name"]].copy()
        road_results["actual"] = y_test.to_numpy()
        road_results["predicted"] = best_pipeline.predict(X_test)
        road_metrics = {}
        for road, group in road_results.groupby("Previous Day - Road/Intersection Name"):
            road_metrics[str(road)] = {
                "records": int(len(group)),
                "mae": float(mean_absolute_error(group["actual"], group["predicted"])),
            }
        all_metrics["targets"][target_key]["road_wise_mae"] = road_metrics

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved model metrics: {METRICS_FILE}")
    print(f"Saved models in: {MODEL_DIR}")
    print("\nImportant: test metrics are a first chronological benchmark. "
          "Do not describe these predictions as live traffic.")

if __name__ == "__main__":
    main()
