from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

import json


st.set_page_config(
    page_title="Rapido Ops Dashboard",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"


@st.cache_data
def load_data():
    """Load booking data for dashboard controls."""
    return pd.read_csv(PROJECT_DIR / "data" / "bookings.csv")


@st.cache_resource
def load_models():
    """Load the models and their input columns."""
    clf = joblib.load(MODEL_DIR / "cancel_model.pkl")
    reg = joblib.load(MODEL_DIR / "fare_model.pkl")
    clf_cols = joblib.load(MODEL_DIR / "cancel_cols.pkl")
    reg_cols = joblib.load(MODEL_DIR / "fare_cols.pkl")

    return clf, reg, clf_cols, reg_cols


clf, reg, clf_cols, reg_cols = load_models()
df = load_data()

# 3. UI Layout
st.title("Rapido Operations Decision Dashboard")
st.caption("Predict cancellations & fare to improve ops decision-making")

tab1, tab2, tab3, tab4, tab5= st.tabs([
    "Cancellation Predictor",
    "Fare Estimator",
    "Feature Importance",
    "EDA",
    "Additional Analysis",
])


with tab1:
    st.header("Cancellation Risk Predictor")
    col1, col2 = st.columns(2)

    with col1:
        hour = st.slider("Hour of Day", 0, 23, 18)
        traffic = st.selectbox("Traffic Level", df['traffic_level'].unique())
        weather = st.selectbox("Weather", df['weather_condition'].unique())

    with col2:
        surge = st.slider("Surge Multiplier", 1.0, 3.0, 1.2, 0.1)
        distance = st.slider("Ride Distance (km)", 1.0, 50.0, 8.0, 0.5)
        vehicle = st.selectbox("Vehicle Type", df['vehicle_type'].unique())

    #if st.button("Predict Cancellation Risk"):
    #    input_df = pd.DataFrame([[hour, traffic, weather, surge, distance, vehicle]],
    #                            columns=['hour_of_day', 'traffic_level', 'weather_condition', 'surge_multiplier', 'ride_distance_km', 'vehicle_type'])
    #    input_encoded = pd.get_dummies(input_df).reindex(columns=clf_cols, fill_value=0)
    #    prob = clf.predict_proba(input_encoded)[0][1] * 100
    #    st.metric("Cancellation Probability", f"{prob:.1f}%")
    #    if prob > 40:
    #        st.error("High risk. Consider driver incentive or reduce ETAs.")
    #    else:
    #        st.success("Low risk. Proceed normally.")

    city = st.selectbox("City", sorted(df["city"].unique()))

    day = st.selectbox(
        "Day of Week",
        [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ],
    )

    is_weekend = int(day in ["Saturday", "Sunday"])

    preferred_vehicle = st.selectbox(
        "Customer Preferred Vehicle",
        sorted(df["vehicle_type"].unique()),
    )

    driver_experience = st.number_input(
        "Driver Experience (years)",
        min_value=0.0,
        value=5.0,
        step=0.5,
    )



    if st.button("Predict Cancellation Risk via API"):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict/cancellation", 

                json={
                    "hour_of_day": hour,
                    "traffic_level": traffic,
                    "weather_condition": weather,
                    "surge_multiplier": surge,
                    "ride_distance_km": distance,
                    "vehicle_type": vehicle,
                    "city": city,
                    "day_of_week": day,
                    "is_weekend": is_weekend,
                    "preferred_vehicle_type": preferred_vehicle,
                    "driver_experience_years": driver_experience,
                },

                timeout=5
            )
            result = response.json()
            prob = result['cancellation_probability']
            
            st.metric("Cancellation Probability", f"{prob}%")
            if prob > 40:
                st.error("High risk. Consider driver incentive or reduce ETAs.")
            else:
                st.success("Low risk. Proceed normally.")
                
        except requests.exceptions.ConnectionError:
                st.error("API not running! Start it with: "
                        "python -m uvicorn api.main:app --reload"
                    )
        except Exception as e:
            st.error(f"Error: {e}")


with tab2:
    st.header("Fare Estimator")
    col1, col2 = st.columns(2)

    with col1:
        f_distance = st.slider("Distance (km)", 1.0, 50.0, 10.0, 0.5, key='fd')
        f_time = st.slider("Estimated Ride Time (min)", 5, 120, 25)
        f_surge = st.slider("Surge Multiplier", 1.0, 3.0, 1.3, 0.1, key='fs')

    with col2:
        f_hour = st.slider("Hour of Day", 0, 23, 19, key='fh')
        f_traffic = st.selectbox("Traffic", df['traffic_level'].unique(), key='ft')
        f_vehicle = st.selectbox("Vehicle", df['vehicle_type'].unique(), key='fv')

    if st.button("Estimate Fare"):
        input_df = pd.DataFrame([[f_distance, f_time, f_surge, f_hour, f_traffic, f_vehicle]],
                                columns=['ride_distance_km', 'estimated_ride_time_min', 'surge_multiplier', 'hour_of_day', 'traffic_level', 'vehicle_type'])
        input_encoded = pd.get_dummies(input_df).reindex(columns=reg_cols, fill_value=0)
        fare = reg.predict(input_encoded)[0]
        st.metric("Predicted Fare", f"₹{fare:.2f}")

with tab3:
    st.header("Feature Importance Ranking")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cancellation Drivers")
        fi_clf = pd.DataFrame({'feature': clf_cols, 'importance': clf.feature_importances_}).sort_values('importance', ascending=False).head(10)
        fig1 = px.bar(fi_clf, x='importance', y='feature', orientation='h', title="Top 10 Features for Cancellation")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Fare Drivers")
        fi_reg = pd.DataFrame({'feature': reg_cols, 'importance': reg.feature_importances_}).sort_values('importance', ascending=False).head(10)
        fig2 = px.bar(fi_reg, x='importance', y='feature', orientation='h', title="Top 10 Features for Fare")
        st.plotly_chart(fig2, use_container_width=True)


with tab4:
    st.header("Ride Patterns and Cancellation Analysis")

    selected_cities = st.multiselect(
        "Filter by City",
        options=sorted(df["city"].unique()),
        default=sorted(df["city"].unique()),
    )

    eda = df[df["city"].isin(selected_cities)].copy()

    if eda.empty:
        st.info("Select at least one city.")
    else:
        eda["is_cancelled"] = (
            eda["booking_status"] == "Cancelled"
        ).astype(int)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Bookings", f"{len(eda):,}")
        col2.metric(
            "Cancelled Bookings",
            f"{eda['is_cancelled'].sum():,}",
        )
        col3.metric(
            "Cancellation Rate",
            f"{eda['is_cancelled'].mean():.1%}",
        )

        # 1. Booking volume by hour
        hourly_volume = (
            eda.groupby("hour_of_day")
            .size()
            .reindex(range(24), fill_value=0)
            .rename("bookings")
            .reset_index()
        )

        st.plotly_chart(
            px.bar(
                hourly_volume,
                x="hour_of_day",
                y="bookings",
                title="Booking Volume by Hour",
            ),
            use_container_width=True,
        )

        # 2. Booking volume by weekday
        weekdays = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]

        weekday_volume = (
            eda.groupby("day_of_week")
            .size()
            .reindex(weekdays, fill_value=0)
            .rename("bookings")
            .reset_index()
        )

        st.plotly_chart(
            px.bar(
                weekday_volume,
                x="day_of_week",
                y="bookings",
                title="Booking Volume by Weekday",
            ),
            use_container_width=True,
        )

        # 3. Booking volume by city
        city_volume = (
            eda.groupby("city")
            .size()
            .rename("bookings")
            .reset_index()
        )

        st.plotly_chart(
            px.bar(
                city_volume,
                x="city",
                y="bookings",
                title="Booking Volume by City",
            ),
            use_container_width=True,
        )

        # 4. Cancellation counts and rates by hour
        hourly_cancellations = (
            eda.groupby("hour_of_day")
            .agg(
                bookings=("booking_id", "size"),
                cancellations=("is_cancelled", "sum"),
                cancellation_rate=("is_cancelled", "mean"),
            )
            .reset_index()
        )

        hourly_cancellations["cancellation_rate_pct"] = (
            hourly_cancellations["cancellation_rate"] * 100
        )

        st.plotly_chart(
            px.bar(
                hourly_cancellations,
                x="hour_of_day",
                y="cancellations",
                title="Cancelled Bookings by Hour",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            px.line(
                hourly_cancellations,
                x="hour_of_day",
                y="cancellation_rate_pct",
                markers=True,
                title="Cancellation Rate by Hour (%)",
                hover_data=["bookings", "cancellations"],
            ),
            use_container_width=True,
        )

        # 5. City-hour cancellation heatmap
        cancellation_heatmap = eda.pivot_table(
            index="city",
            columns="hour_of_day",
            values="is_cancelled",
            aggfunc="mean",
        ).reindex(columns=range(24)) * 100

        st.plotly_chart(
            px.imshow(
                cancellation_heatmap,
                aspect="auto",
                color_continuous_scale="Reds",
                labels={
                    "x": "Hour of Day",
                    "y": "City",
                    "color": "Cancellation %",
                },
                title="Cancellation Rate by City and Hour",
            ),
            use_container_width=True,
        )

        # Download the aggregated results.
        st.download_button(
            "Download Hourly Cancellation Summary",
            data=hourly_cancellations.to_csv(index=False),
            file_name="hourly_cancellation_summary.csv",
            mime="text/csv",
        )



        # 6. Distance versus fare for completed rides
        completed_eda = eda[
            eda["booking_status"] == "Completed"
        ].copy()

        st.subheader("Distance and Fare")

        if len(completed_eda) > 1:
            correlation = completed_eda["ride_distance_km"].corr(
                completed_eda["booking_value"]
            )

            st.metric("Distance–Fare Correlation", f"{correlation:.3f}")

            # Limit displayed points; correlation uses all completed rides.
            plot_sample = completed_eda.sample(
                n=min(3000, len(completed_eda)),
                random_state=42,
            )

            st.plotly_chart(
                px.scatter(
                    plot_sample,
                    x="ride_distance_km",
                    y="booking_value",
                    color="vehicle_type",
                    opacity=0.4,
                    labels={
                        "ride_distance_km": "Distance (km)",
                        "booking_value": "Fare (₹)",
                    },
                    title="Distance versus Fare — Completed Rides",
                ),
                use_container_width=True,
            )
        else:
            st.info("Not enough completed rides for correlation analysis.")

        # 7. Traffic and weather versus cancellation
        st.subheader("Traffic, Weather and Cancellations")

        for column, label in [
            ("traffic_level", "Traffic Level"),
            ("weather_condition", "Weather"),
        ]:
            summary = (
                eda.groupby(column)
                .agg(
                    bookings=("booking_id", "size"),
                    cancellations=("is_cancelled", "sum"),
                    cancellation_rate=("is_cancelled", "mean"),
                )
                .reset_index()
            )

            summary["cancellation_rate_pct"] = (
                summary["cancellation_rate"] * 100
            )

            st.plotly_chart(
                px.bar(
                    summary,
                    x=column,
                    y="cancellation_rate_pct",
                    hover_data=["bookings", "cancellations"],
                    labels={
                        column: label,
                        "cancellation_rate_pct": "Cancellation Rate (%)",
                    },
                    title=f"Cancellation Rate by {label}",
                ),
                use_container_width=True,
            )

        # 8. Average surge by hour
        surge_summary = (
            eda.groupby("hour_of_day")
            .agg(
                average_surge=("surge_multiplier", "mean"),
                bookings=("booking_id", "size"),
            )
            .reset_index()
        )

        st.plotly_chart(
            px.line(
                surge_summary,
                x="hour_of_day",
                y="average_surge",
                markers=True,
                hover_data=["bookings"],
                labels={
                    "hour_of_day": "Hour of Day",
                    "average_surge": "Average Surge Multiplier",
                },
                title="Average Surge by Hour",
            ),
            use_container_width=True,
        )

        # 9. Customer and driver average rating distributions
        st.subheader("Customer and Driver Ratings")

        customers_eda = pd.read_csv(
            PROJECT_DIR / "data" / "customers.csv"
        )
        drivers_eda = pd.read_csv(
            PROJECT_DIR / "data" / "drivers.csv"
        )

        # Include each person once, based on the filtered bookings.
        customers_eda = customers_eda[
            customers_eda["customer_id"].isin(eda["customer_id"])
        ].drop_duplicates("customer_id")

        drivers_eda = drivers_eda[
            drivers_eda["driver_id"].isin(eda["driver_id"])
        ].drop_duplicates("driver_id")

        st.caption(
            "Ratings are person-level averages for customers and drivers "
            "appearing in the selected bookings, not individual ride ratings."
        )

        rating_col1, rating_col2 = st.columns(2)

        with rating_col1:
            st.plotly_chart(
                px.histogram(
                    customers_eda,
                    x="avg_customer_rating",
                    nbins=20,
                    labels={
                        "avg_customer_rating": "Average Customer Rating",
                    },
                    title="Customer Average Rating Distribution",
                ),
                use_container_width=True,
            )

        with rating_col2:
            st.plotly_chart(
                px.histogram(
                    drivers_eda,
                    x="avg_driver_rating",
                    nbins=20,
                    labels={
                        "avg_driver_rating": "Average Driver Rating",
                    },
                    title="Driver Average Rating Distribution",
                ),
                use_container_width=True,
            )

with tab5:
    st.header("Locations, Behaviour and Additional Models")

    report_dir = PROJECT_DIR / "reports"

    required_files = [
        "enriched_bookings.csv",
        "location_demand_summary.csv",
        "customer_scores.csv",
        "driver_scores.csv",
        "additional_metrics.json",
    ]

    if not all((report_dir / name).exists() for name in required_files):
        st.info(
            "Run python src/complete_remaining.py first."
        )
    else:
        enriched = pd.read_csv(
            report_dir / "enriched_bookings.csv"
        )
        demand = pd.read_csv(
            report_dir / "location_demand_summary.csv"
        )
        customer_scores = pd.read_csv(
            report_dir / "customer_scores.csv"
        )
        driver_scores = pd.read_csv(
            report_dir / "driver_scores.csv"
        )

        with (report_dir / "additional_metrics.json").open(
            encoding="utf-8"
        ) as file:
            extra_metrics = json.load(file)

        city_choice = st.selectbox(
            "Analysis City",
            sorted(enriched["city"].unique()),
            key="additional_city",
        )

        city_bookings = enriched[
            enriched["city"] == city_choice
        ].copy()

        # Location heatmaps use location IDs, not geographic coordinates.
        st.subheader("Pickup and Drop Heatmaps")
        st.caption(
            "These are location-category heatmaps, not geographic maps."
        )

        for location_column, title in [
            ("pickup_location", "Pickup"),
            ("drop_location", "Drop"),
        ]:
            volume = pd.crosstab(
                city_bookings[location_column],
                city_bookings["hour_of_day"],
            ).reindex(columns=range(24), fill_value=0)

            st.plotly_chart(
                px.imshow(
                    volume,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    labels={
                        "x": "Hour",
                        "y": f"{title} Location",
                        "color": "Bookings",
                    },
                    title=f"{title} Volume by Location and Hour",
                ),
                use_container_width=True,
            )

        # Customer/driver comparison: distinct measures, clearly labelled.
        st.subheader("Customer and Driver Behaviour")

        city_customers = customer_scores[
            customer_scores["customer_id"].isin(
                city_bookings["customer_id"]
            )
        ]

        city_drivers = driver_scores[
            driver_scores["driver_id"].isin(
                city_bookings["driver_id"]
            )
        ]

        behaviour = pd.DataFrame([
            {
                "Measure": "Customer cancellation rate",
                "Percentage": (
                    city_customers["cancelled_rides"].sum()
                    / max(city_customers["total_bookings"].sum(), 1)
                    * 100
                ),
            },
            {
                "Measure": "Driver acceptance rate",
                "Percentage": (
                    city_drivers["accepted_rides"].sum()
                    / max(
                        city_drivers["total_assigned_rides"].sum(), 1
                    )
                    * 100
                ),
            },
            {
                "Measure": "Driver delay rate",
                "Percentage": (
                    city_drivers["delay_count"].sum()
                    / max(
                        city_drivers["total_assigned_rides"].sum(), 1
                    )
                    * 100
                ),
            },
        ])

        st.caption(
            "Full-dataset history for people appearing in this city's "
            "bookings. These are different measures, not equivalent rates."
        )

        st.plotly_chart(
            px.bar(
                behaviour,
                x="Measure",
                y="Percentage",
                title="Customer Cancellation and Driver Service Measures",
            ),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Mean Customer Loyalty Score",
            f"{city_customers['customer_loyalty_score'].mean():.1f}/100",
        )
        col2.metric(
            "Mean Driver Reliability Score",
            f"{city_drivers['driver_reliability_score'].mean():.1f}/100",
        )

        st.caption(
            "Loyalty = completed bookings / total bookings × 100. "
            "Reliability = acceptance rate × (1 − delay rate) × 100. "
            "Both are descriptive project scores."
        )

        # Integrate the supplied demand data into descriptive analysis.
        st.subheader("Location Demand")

        city_demand = demand[demand["city"] == city_choice]

        demand_heatmap = city_demand.pivot_table(
            index="pickup_location",
            columns="hour_of_day",
            values="total_requests",
            aggfunc="sum",
        )

        st.plotly_chart(
            px.imshow(
                demand_heatmap,
                aspect="auto",
                color_continuous_scale="Oranges",
                labels={
                    "x": "Hour",
                    "y": "Pickup Location",
                    "color": "Requests",
                },
                title="Requests from the Location Demand Dataset",
            ),
            use_container_width=True,
        )

        # Display engineered and time-enriched features.
        st.subheader("Engineered Features")

        st.dataframe(
            city_bookings[[
                "booking_id",
                "city_pair",
                "rush_hour_flag",
                "long_distance_flag",
                "peak_time_flag",
                "season",
                "fare_per_km",
                "fare_per_min",
            ]].head(100),
            hide_index=True,
        )

        st.caption(
            "Fare ratios are descriptive and are excluded from prediction "
            "inputs because they depend on the actual fare."
        )

        # Prediction demonstration using an existing booking.
        st.subheader("Additional Model Demonstration")

        st.caption(
            "This demonstrates saved-model execution using a dataset "
            "record. Accuracy is assessed separately in the test reports."
        )

        booking_id = st.selectbox(
            "Choose a Booking",
            city_bookings["booking_id"].tolist(),
        )

        if st.button("Run Additional Models"):
            row = city_bookings[
                city_bookings["booking_id"] == booking_id
            ].head(1)

            outcome_artifact = joblib.load(
                MODEL_DIR / "ride_outcome.pkl"
            )

            predicted_outcome = outcome_artifact["pipeline"].predict(
                row[outcome_artifact["features"]]
            )[0]

            st.write(
                f"**Predicted ride outcome:** {predicted_outcome}"
            )

            # Fare model is trained on completed rides.
            fare_artifact = joblib.load(
                MODEL_DIR / "fare_tuned.pkl"
            )

            fare = fare_artifact["pipeline"].predict(
                row[fare_artifact["features"]]
            )[0]

            st.metric("Tuned Fare Estimate", f"₹{fare:.2f}")

            customer_row = customer_scores[
                customer_scores["customer_id"]
                == row["customer_id"].iloc[0]
            ].head(1)

            driver_row = driver_scores[
                driver_scores["driver_id"]
                == row["driver_id"].iloc[0]
            ].head(1)

            for name, person_row, label in [
                (
                    "customer_risk",
                    customer_row,
                    "Customer Risk-Flag Estimate",
                ),
                (
                    "driver_delay",
                    driver_row,
                    "Driver Delay-Flag Estimate",
                ),
            ]:
                if person_row.empty:
                    st.info(f"No profile found for {label}.")
                    continue

                artifact = joblib.load(
                    MODEL_DIR / f"{name}.pkl"
                )
                pipeline = artifact["pipeline"]

                positive_index = list(
                    pipeline.named_steps["model"].classes_
                ).index(1)

                probability = pipeline.predict_proba(
                    person_row[artifact["features"]]
                )[0, positive_index]

                st.metric(label, f"{probability:.1%}")

            st.caption(
                "Profile models estimate supplied snapshot flags. "
                "They have not established future booking risk."
            )

        st.subheader("Additional Evaluation Results")
        st.json(extra_metrics)