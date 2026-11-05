from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import json

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# Locate folders relative to this script.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"


def train_models():
    """Train and save the cancellation and fare models."""
    df = pd.read_csv(DATA_DIR / "bookings.csv")

    customers = pd.read_csv(
        DATA_DIR / "customers.csv",
        usecols=["customer_id", "preferred_vehicle_type"],
    )

    drivers = pd.read_csv(
        DATA_DIR / "drivers.csv",
        usecols=["driver_id", "driver_experience_years"],
    )

    df = df.merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    df = df.merge(
        drivers,
        on="driver_id",
        how="left",
        validate="many_to_one",
    )

    df["preferred_vehicle_type"] = df["preferred_vehicle_type"].fillna("Unknown")
    df["driver_experience_years"] = df["driver_experience_years"].fillna(0)


    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Cancellation model: Incomplete rides are not labelled Cancelled.
    df["is_cancelled"] = (df["booking_status"] == "Cancelled").astype(int)

    cancellation_features = [
        "hour_of_day",
        "traffic_level",
        "weather_condition",
        "surge_multiplier",
        "ride_distance_km",
        "vehicle_type",
        "city",
        "day_of_week",
        "is_weekend",
        "preferred_vehicle_type",
        "driver_experience_years",
    ]

    X_clf = pd.get_dummies(
        df[cancellation_features], drop_first=True
    )
    y_clf = df["is_cancelled"]

    if y_clf.nunique() != 2:
        raise ValueError(
            "Check booking_status: expected Cancelled and non-cancelled rides."
        )

    clf = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight="balanced",
    )

    # Use 80% for training and 20% for testing.
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf,
        y_clf,
        test_size=0.2,
        random_state=42,
        stratify=y_clf,
    )


    # Compare four combinations using three validation folds.
    parameter_grid = {
        "max_depth": [10, None],
        "min_samples_leaf": [1, 5],
    }

    cross_validation = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cross_validation,
        n_jobs=2,
        verbose=1,
    )

    grid_search.fit(X_clf_train, y_clf_train)

    # Use the best model, already refitted on the training set.
    clf = grid_search.best_estimator_

    print("\nBest cancellation model settings:")
    print(grid_search.best_params_)
    print(f"Best cross-validation F1: {grid_search.best_score_:.4f}")


    report_dir = PROJECT_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(grid_search.cv_results_).to_csv(
        report_dir / "cancellation_grid_search.csv",
        index=False,
    )


    clf_predictions = clf.predict(X_clf_test)
    clf_probabilities = clf.predict_proba(X_clf_test)[:, 1]

    classification_metrics = {
        "accuracy": float(accuracy_score(y_clf_test, clf_predictions)),
        "f1_score": float(f1_score(y_clf_test, clf_predictions, zero_division=0)),
        "auc": float(roc_auc_score(y_clf_test, clf_probabilities)),
        "confusion_matrix": confusion_matrix(
            y_clf_test, clf_predictions, labels=[0, 1]
        ).tolist(),
    }

    print("\nCancellation model — test results")
    print(json.dumps(classification_metrics, indent=4))


    # Fare model: train only on completed rides.
    completed = df[df["booking_status"] == "Completed"].copy()

    fare_features = [
        "ride_distance_km",
        "estimated_ride_time_min",
        "surge_multiplier",
        "hour_of_day",
        "traffic_level",
        "vehicle_type",
    ]

    X_reg = pd.get_dummies(
        completed[fare_features], drop_first=True
    )
    y_reg = completed["booking_value"]

    reg = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
    )

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg,
        y_reg,
        test_size=0.2,
        random_state=42,
    )

    reg.fit(X_reg_train, y_reg_train)

    fare_predictions = reg.predict(X_reg_test)

    regression_metrics = {
        "rmse": float(mean_squared_error(y_reg_test, fare_predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_reg_test, fare_predictions)),
        "r2": float(r2_score(y_reg_test, fare_predictions)),
    }

    print("\nFare model — test results")
    print(json.dumps(regression_metrics, indent=4))

    # Save both models' evaluation results.
    report_dir = PROJECT_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    with (report_dir / "evaluation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "cancellation": classification_metrics,
                "fare": regression_metrics,
            },
            file,
            indent=4,
        )
   

    # Save models and their input column names.
    joblib.dump(clf, MODEL_DIR / "cancel_model.pkl")
    joblib.dump(reg, MODEL_DIR / "fare_model.pkl")
    joblib.dump(X_clf.columns.tolist(), MODEL_DIR / "cancel_cols.pkl")
    joblib.dump(X_reg.columns.tolist(), MODEL_DIR / "fare_cols.pkl")

    print(f"Training complete. Four files saved in: {MODEL_DIR}")


if __name__ == "__main__":
    train_models()