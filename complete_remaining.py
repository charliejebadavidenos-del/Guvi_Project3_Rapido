"""Train additional Rapido models and prepare analysis datasets."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"


def make_pipeline(X, estimator):
    """Fit missing-value handling and encoding within training."""
    categorical = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical = [
        column for column in X.columns if column not in categorical
    ]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numerical,
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]),
                categorical,
            ),
        ]
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("model", estimator),
    ])


def prepare_bookings(bookings, customers, drivers, time_features):
    """Join profile and time fields and create booking features."""
    df = bookings.copy()

    df = df.merge(
        customers[["customer_id", "preferred_vehicle_type"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    df = df.merge(
        drivers[["driver_id", "driver_experience_years"]],
        on="driver_id",
        how="left",
        validate="many_to_one",
    )

    # Join on the booking's calendar date and hour.
    df["booking_hour"] = (
        pd.to_datetime(df["booking_date"], errors="coerce").dt.normalize()
        + pd.to_timedelta(df["hour_of_day"], unit="h")
    )

    if df["booking_hour"].isna().any():
        raise ValueError("Some booking dates could not be parsed.")

    times = time_features.copy()
    times["booking_hour"] = pd.to_datetime(
        times["datetime"], errors="coerce"
    ).dt.floor("h")

    if times["booking_hour"].isna().any():
        raise ValueError("Some time-feature dates could not be parsed.")

    if times["booking_hour"].duplicated().any():
        raise ValueError("time_features has duplicate hourly timestamps.")

    time_columns = [
        "booking_hour", "is_holiday", "peak_time_flag", "season"
    ]

    df = df.merge(
        times[time_columns],
        on="booking_hour",
        how="left",
        validate="many_to_one",
    )

    if df[["is_holiday", "peak_time_flag", "season"]].isna().any().any():
        raise ValueError("Some bookings have no matching time features.")

    df["rush_hour_flag"] = df["hour_of_day"].isin(
        [8, 9, 10, 17, 18, 19, 20]
    ).astype(int)

    # Project definition: longer than 15 km.
    df["long_distance_flag"] = (
        df["ride_distance_km"] > 15
    ).astype(int)

    df["city_pair"] = (
        df["city"].astype(str)
        + ": "
        + df["pickup_location"].astype(str)
        + " -> "
        + df["drop_location"].astype(str)
    )

    # Descriptive features only: booking_value is unknown before prediction.
    df["fare_per_km"] = (
        df["booking_value"]
        / df["ride_distance_km"].where(df["ride_distance_km"] > 0)
    )

    df["fare_per_min"] = (
        df["booking_value"]
        / df["estimated_ride_time_min"].where(
            df["estimated_ride_time_min"] > 0
        )
    )

    return df


def train_classifier(name, frame, features, target, binary):
    """Train and evaluate a classifier on an 80/20 split."""
    data = frame.dropna(subset=[target]).copy()

    X = data[features]
    y = data[target].astype(int) if binary else data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = make_pipeline(
        X,
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=2,
        ),
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    classes = pipeline.named_steps["model"].classes_

    average = "binary" if binary else "macro"

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                average=average,
                zero_division=0,
            )
        ),
        "classes": classes.tolist(),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=classes
        ).tolist(),
    }

    if binary:
        positive_index = list(classes).index(1)
        metrics["auc"] = float(
            roc_auc_score(y_test, probabilities[:, positive_index])
        )
    else:
        metrics["auc_macro_ovr"] = float(
            roc_auc_score(
                y_test,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        )

    joblib.dump(
        {"pipeline": pipeline, "features": features},
        MODEL_DIR / f"{name}.pkl",
    )

    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions,
    }).to_csv(
        REPORT_DIR / f"{name}_test_predictions.csv",
        index=False,
    )

    print(f"\n{name}")
    print(json.dumps(metrics, indent=4))

    return metrics


def tune_fare(df):
    """Tune a fare pipeline using cross-validation on training data."""
    completed = df[df["booking_status"] == "Completed"].copy()

    features = [
        "ride_distance_km",
        "estimated_ride_time_min",
        "surge_multiplier",
        "hour_of_day",
        "traffic_level",
        "weather_condition",
        "vehicle_type",
        "city",
        "is_weekend",
        "rush_hour_flag",
        "long_distance_flag",
        "is_holiday",
        "peak_time_flag",
        "season",
    ]

    X = completed[features]
    y = completed["booking_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = make_pipeline(
        X,
        RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=1,
        ),
    )

    search = GridSearchCV(
        pipeline,
        param_grid={
            "model__max_depth": [15, None],
            "model__min_samples_leaf": [1, 3],
        },
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=2,
        verbose=1,
    )

    search.fit(X_train, y_train)
    predictions = search.best_estimator_.predict(X_test)

    rmse = float(mean_squared_error(y_test, predictions) ** 0.5)

    metrics = {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "rmse_pct_of_mean_fare": float(
            rmse / y_test.mean() * 100
        ),
        "best_cv_rmse": float(-search.best_score_),
        "best_parameters": search.best_params_,
    }

    joblib.dump(
        {
            "pipeline": search.best_estimator_,
            "features": features,
        },
        MODEL_DIR / "fare_tuned.pkl",
    )

    pd.DataFrame(search.cv_results_).to_csv(
        REPORT_DIR / "fare_grid_search.csv",
        index=False,
    )

    print("\nTuned fare model")
    print(json.dumps(metrics, indent=4))

    return metrics


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    bookings = pd.read_csv(DATA_DIR / "bookings.csv")
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    drivers = pd.read_csv(DATA_DIR / "drivers.csv")
    time_features = pd.read_csv(DATA_DIR / "time_features.csv")
    demand = pd.read_csv(DATA_DIR / "location_demand.csv")

    enriched = prepare_bookings(
        bookings, customers, drivers, time_features
    )

    # Save analysis fields for the dashboard.
    enriched.to_csv(
        REPORT_DIR / "enriched_bookings.csv",
        index=False,
    )

    # Demand is descriptive: its aggregate counts may include future rides.
    # Do not use those counts as pre-booking model predictors.
    demand["cancellation_rate_pct"] = (
        demand["cancelled_rides"]
        / demand["total_requests"].where(demand["total_requests"] > 0)
        * 100
    )

    demand.to_csv(
        REPORT_DIR / "location_demand_summary.csv",
        index=False,
    )

    # Transparent descriptive scores, not additional trained models.
    # These describe the supplied snapshot, not future reliability.
    drivers["driver_reliability_score"] = (
        100
        * drivers["acceptance_rate"].clip(0, 1)
        * (1 - drivers["delay_rate"].clip(0, 1))
    )

    customers["customer_loyalty_score"] = (
        100
        * customers["completed_rides"]
        / customers["total_bookings"].where(
            customers["total_bookings"] > 0
        )
    ).clip(0, 100)

    drivers.to_csv(
        REPORT_DIR / "driver_scores.csv",
        index=False,
    )

    customers.to_csv(
        REPORT_DIR / "customer_scores.csv",
        index=False,
    )

    booking_features = [
        "hour_of_day",
        "traffic_level",
        "weather_condition",
        "surge_multiplier",
        "ride_distance_km",
        "estimated_ride_time_min",
        "vehicle_type",
        "city",
        "day_of_week",
        "is_weekend",
        "preferred_vehicle_type",
        "driver_experience_years",
        "rush_hour_flag",
        "long_distance_flag",
        "is_holiday",
        "peak_time_flag",
        "season",
    ]

    results = {}

    results["ride_outcome"] = train_classifier(
        name="ride_outcome",
        frame=enriched,
        features=booking_features,
        target="booking_status",
        binary=False,
    )

    # Exclude rates, counts, ratings of uncertain timing, IDs and flags.
    results["customer_risk"] = train_classifier(
        name="customer_risk",
        frame=customers,
        features=[
            "customer_age",
            "customer_city",
            "customer_signup_days_ago",
            "preferred_vehicle_type",
        ],
        target="customer_cancel_flag",
        binary=True,
    )

    results["driver_delay"] = train_classifier(
        name="driver_delay",
        frame=drivers,
        features=[
            "driver_age",
            "driver_city",
            "vehicle_type",
            "driver_experience_years",
        ],
        target="driver_delay_flag",
        binary=True,
    )

    results["fare_tuned"] = tune_fare(enriched)

    results["limitations"] = [
        "Customer and driver models predict supplied snapshot flags.",
        "Future cancellation and delay performance is not established.",
        "Aggregate demand and descriptive scores are excluded from predictors.",
        "Fare-per-km and fare-per-minute are analysis-only features.",
        "Random holdouts provide exploratory evaluation, not temporal validation.",
        "Existing holdouts have been inspected during development.",
    ]

    with (REPORT_DIR / "additional_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    print("\nComplete: additional models and reports saved.")


if __name__ == "__main__":
    main()