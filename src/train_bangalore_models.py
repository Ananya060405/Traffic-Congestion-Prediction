"""Train and evaluate Bengaluru next-day traffic prediction models.

Models use previous-day observations and known calendar features to predict
next-day congestion and traffic volume. Splits are chronological by target date.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

def regression_metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {DATA_FILE}\n"
            "Run: python src/prepare_bangalore_data.py"
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["Date", "Target Date"])
    df = df.sort_values(["Target Date", "Previous Day - Road/Intersection Name"]).reset_index(drop=True)
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} training examples found; refusing to train.")

    # Split on unique dates so records from the same target day cannot leak
    # across train, validation, and test partitions.
    unique_dates = pd.Index(sorted(df["Target Date"].dropna().unique()))
    if len(unique_dates) < 20:
        raise ValueError("Not enough distinct target dates for chronological evaluation.")
    train_end = unique_dates[int(len(unique_dates) * 0.70)]
    validation_end = unique_dates[int(len(unique_dates) * 0.85)]

    train_df = df[df["Target Date"] < train_end].copy()
    validation_df = df[
        (df["Target Date"] >= train_end) & (df["Target Date"] < validation_end)
    ].copy()
    test_df = df[df["Target Date"] >= validation_end].copy()

    if min(len(train_df), len(validation_df), len(test_df)) == 0:
        raise ValueError("A chronological split is empty; inspect the date coverage.")

    X_train = train_df[FEATURES]
    X_validation = validation_df[FEATURES]
    X_test = test_df[FEATURES]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_metrics = {
        "dataset": str(DATA_FILE.relative_to(ROOT)),
        "examples": {
            "train": int(len(train_df)),
            "validation": int(len(validation_df)),
            "test": int(len(test_df)),
        },
        "target_date_ranges": {
            "train": [str(train_df["Target Date"].min().date()), str(train_df["Target Date"].max().date())],
            "validation": [str(validation_df["Target Date"].min().date()), str(validation_df["Target Date"].max().date())],
            "test": [str(test_df["Target Date"].min().date()), str(test_df["Target Date"].max().date())],
        },
        "prediction_scope": "next calendar day, using previous calendar day's observed features",
        "targets": {},
    }

    for target_key, target_column in TARGETS.items():
        y_train = train_df[target_column].astype(float)
        y_validation = validation_df[target_column].astype(float)
        y_test = test_df[target_column].astype(float)

        candidates = []
        for model_name, estimator in model_candidates().items():
            candidate = Pipeline([
                ("preprocessor", make_preprocessor()),
                ("model", estimator),
            ])
            candidate.fit(X_train, y_train)
            validation_prediction = candidate.predict(X_validation)
            validation_score = regression_metrics(y_validation, validation_prediction)
            candidates.append((validation_score["mae"], model_name, estimator, validation_score))
            print(
                f"{target_key:14s} | {model_name:18s} | "
                f"validation MAE {validation_score['mae']:.3f} | "
                f"RMSE {validation_score['rmse']:.3f} | R2 {validation_score['r2']:.4f}"
            )

        # Select on validation only, then refit on train + validation.
        candidates.sort(key=lambda item: item[0])
        _, best_name, best_estimator, best_validation_metrics = candidates[0]
        fit_df = pd.concat([train_df, validation_df], ignore_index=True)
        final_pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", best_estimator),
        ])
        final_pipeline.fit(fit_df[FEATURES], fit_df[target_column].astype(float))

        test_prediction = final_pipeline.predict(X_test)
        test_metrics = regression_metrics(y_test, test_prediction)
        model_path = MODEL_DIR / f"bangalore_{target_key}_model.joblib"
        joblib.dump(final_pipeline, model_path)

        road_results = test_df[["Previous Day - Road/Intersection Name"]].copy()
        road_results["actual"] = y_test.to_numpy()
        road_results["predicted"] = test_prediction
        road_metrics = {}
        for road, group in road_results.groupby("Previous Day - Road/Intersection Name"):
            road_metrics[str(road)] = {
                "records": int(len(group)),
                "mae": float(mean_absolute_error(group["actual"], group["predicted"])),
            }

        all_metrics["targets"][target_key] = {
            "selected_model": best_name,
            "validation_metrics": best_validation_metrics,
            "test_metrics": test_metrics,
            "model_path": str(model_path.relative_to(ROOT)),
            "validation_model_comparison": [
                {"model": name, **score}
                for _, name, _, score in candidates
            ],
            "road_wise_test_mae": road_metrics,
        }
        print(
            f"  FINAL TEST | {target_key:14s} | "
            f"MAE {test_metrics['mae']:.3f} | "
            f"RMSE {test_metrics['rmse']:.3f} | R2 {test_metrics['r2']:.4f}"
        )

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved model metrics: {METRICS_FILE}")
    print(f"Saved models in: {MODEL_DIR}")
    print("\nImportant: these are next-day predictions from historical data, not live traffic.")

if __name__ == "__main__":
    main()
