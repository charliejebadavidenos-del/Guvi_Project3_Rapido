# Rapido: Intelligent Mobility Insights

A Python project for analysing ride patterns, predicting cancellations
and ride outcomes, estimating fares, and demonstrating customer and
driver risk classification.

## Features

- Interactive Streamlit dashboard with city filters.
- Booking volumes by hour, weekday and city.
- Cancellation counts, rates and city/hour heatmaps.
- Pickup/drop location heatmaps.
- Distance/fare, weather, traffic, surge and rating analysis.
- Customer and driver behaviour summaries.
- Binary cancellation prediction through FastAPI.
- Three-class ride outcome prediction.
- Customer cancellation-risk flag classification.
- Driver delay-flag classification.
- Fare prediction and hyperparameter tuning.
- Evaluation reports and CSV downloads.

## Project Structure

| Path | Purpose |
|---|---|
| app/dashboard.py | Streamlit dashboard |
| api/main.py | FastAPI prediction endpoints |
| src/train_models.py | Original cancellation/fare training and evaluation |
| src/complete_remaining.py | Additional models, features and fare tuning |
| data/ | Five input CSV datasets |
| models/ | Generated models and feature-column files |
| notebooks/ | Original data inspection and MySQL loading notebook |
| reports/ | Evaluation results and prepared analysis datasets |
| requirements.txt | Installed package versions |

## Data

Place these files inside data/:

- bookings.csv
- customers.csv
- drivers.csv
- location_demand.csv
- time_features.csv

The supplied data contains 100,000 bookings, 10,000 customers and
5,000 drivers.

## Setup on Windows

Run commands from the project root.

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Use the same Python version as the development environment:
Python 3.14. Package versions are recorded in requirements.txt.

## Train Models

```bat
python src/train_models.py
python src/complete_remaining.py
```

These commands generate models in models/ and reports in reports/.
Run both scripts before opening the full dashboard.

## Run the API

In an activated terminal:

```bat
python -m uvicorn api.main:app --reload
```

API documentation: http://127.0.0.1:8000/docs

Endpoints:

- GET /
- POST /predict/cancellation
- POST /predict/fare

## Run the Dashboard

In a second activated terminal:

```bat
python -m streamlit run app/dashboard.py
```

Open http://localhost:8501

The API must remain running for the dashboard's API cancellation button.

## Dashboard Walkthrough

1. Cancellation Predictor: enter booking conditions and request a prediction.
2. Fare Estimator: estimate fare using the original saved fare model.
3. Feature Importance: inspect the original models' feature rankings.
4. EDA: explore ride patterns, filters and downloadable summaries.
5. Additional Analysis: inspect location demand, engineered features,
   behaviour summaries, additional models and evaluation results.

The tuned fare model is demonstrated in Additional Analysis.
The original Fare Estimator and API use fare_model.pkl.

## Methods

- Random Forest classification and regression.
- 80/20 training/test splits.
- Stratified splits for classification.
- Three-fold GridSearchCV for the original cancellation model and
  the additional fare model.
- Classification accuracy, F1, AUC and confusion matrices.
- Regression RMSE, MAE and R².
- Preprocessing pipelines for the additional models, fitted on training data.

Customer cancellation totals and rates describe the same booking dataset.
They are excluded from model predictors to avoid target leakage.
Aggregate demand is used descriptively rather than as a future demand signal.

## Recorded Test Results

| Model | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| Binary cancellation | 59.995% | 0.4784 | 0.7024 |
| Three-class ride outcome | 48.795% | 0.4117 macro | 0.6685 macro OvR |
| Customer risk flag | 51.35% | 0.5329 | 0.5151 |
| Driver delay flag | 65.10% | 0.2014 | 0.4737 |

| Fare Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Original | ₹12.61 | ₹8.94 | 0.9960 |
| Tuned | ₹11.98 | ₹8.57 | 0.9964 |

Tuned fare RMSE is approximately 3.70% of the mean test fare.
This does not mean every prediction is within 10% of its actual fare.

## Limitations

- Classification accuracy does not meet the brief's 85–90% target.
- Customer and driver models predict supplied snapshot flags.
  Future cancellation and delay prediction is not established.
- Customer and driver AUC results show little useful discrimination.
- Accuracy is below majority-class baselines for these classifiers.
  Other metrics and error tradeoffs must also be considered.
- Random holdouts provide exploratory evaluation rather than future-period
  validation. Test results were inspected during development.
- Probability outputs have not been calibrated.
- Classification test metrics use model predictions at a 50% threshold;
  the cancellation dashboard uses a separate 40% risk-alert threshold.
- Geographic coordinates are unavailable. Location heatmaps use category IDs.
- Payment-method and separate customer/driver cancellation-reason fields
  are unavailable.
- Fare ratios, loyalty and reliability scores are descriptive features.
- Cancellation reduction, ETA improvements and operational business gains
  have not been measured in a live experiment.

## MySQL Notebook

The original notebook documents dataset inspection and MySQL loading.
The current training scripts and dashboard read local CSV files.
MySQL is therefore not required for their execution.

To rerun the notebook, configure local database credentials and file paths.
Do not commit passwords or private connection settings.

## Future Improvements

- Collect timestamped, pre-booking customer and driver history.
- Evaluate on a later time period and compare against simple baselines.
- Improve classification features and probability calibration.
- Select operational thresholds using validation data and business costs.
- Integrate the tuned fare pipeline into the main API and Fare Estimator.