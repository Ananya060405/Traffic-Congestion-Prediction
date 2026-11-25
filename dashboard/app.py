
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

@st.cache_resource
def load_models():

    # Load trained models
    original_lstm = tf.keras.models.load_model(
        "models/traffic_lstm.keras"
    )

    multivariate_lstm = tf.keras.models.load_model(
        "models/traffic_multivariate_lstm.keras"
    )

    transformer = tf.keras.models.load_model(
        "models/traffic_transformer.keras"
    )

    # Load test datasets
    uni_data = np.load("data/processed/lstm_data.npz")
    multi_data = np.load("data/processed/multivariate_lstm_data.npz")

    # Load scalers
    uni_scaler = joblib.load("models/traffic_scaler.pkl")

    multi_scaler = joblib.load(
        "models/multivariate_target_scaler.pkl"
    )

    return (
        original_lstm,
        multivariate_lstm,
        transformer,
        uni_data,
        multi_data,
        uni_scaler,
        multi_scaler
    )

# Page configuration
st.set_page_config(
    page_title="Traffic Congestion Prediction",
    page_icon="🚦",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
    color: #1E88E5;
}
.subtitle {
    color: gray;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<p class="main-title">🚦 Traffic Congestion Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI-powered traffic volume prediction using Deep Learning</p>',
    unsafe_allow_html=True
)

st.divider()

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    ["Dashboard", "Traffic Prediction", "Model Comparison", "About Project"]
)

# Dashboard page
if page == "Dashboard":

    st.header("Traffic Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", "40,575")
    col2.metric("Models Trained", "3")
    col3.metric("Prediction Horizon", "Next 1 Hour")

    st.divider()

    st.subheader("Project Information")

    st.write("""
    This application uses deep learning models to predict
    traffic volume based on historical traffic and weather data.

    **Models available:**
    - Original LSTM
    - Multivariate LSTM
    - Transformer
    """)

    st.info(
        "Currently using historical Metro Interstate Traffic Volume data."
    )
    
    
    st.divider()

    st.subheader("📈 Historical Traffic Analysis")

    # Load cleaned historical dataset
    try:
        traffic_df = pd.read_csv(
            "data/processed/traffic_cleaned.csv"
        )

        traffic_df["date_time"] = pd.to_datetime(
            traffic_df["date_time"]
        )

        # Select date range
        min_date = traffic_df["date_time"].min().date()
        max_date = traffic_df["date_time"].max().date()

        start_date = st.date_input(
            "Select starting date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )

        end_date = st.date_input(
            "Select ending date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

        if start_date > end_date:
            st.error("Starting date cannot be after ending date.")

        else:
            filtered_df = traffic_df[
                (traffic_df["date_time"].dt.date >= start_date) &
                (traffic_df["date_time"].dt.date <= end_date)
            ]

            if filtered_df.empty:
                st.warning("No traffic records found for this date range.")

            else:
                # Traffic trend
                st.subheader("Traffic Volume Over Time")

                chart_df = filtered_df.set_index("date_time")[
                    ["traffic_volume"]
                ]

                st.line_chart(chart_df)

                # Summary metrics
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Average Traffic",
                    f"{filtered_df['traffic_volume'].mean():.0f}"
                )

                col2.metric(
                    "Maximum Traffic",
                    f"{filtered_df['traffic_volume'].max():.0f}"
                )

                col3.metric(
                    "Minimum Traffic",
                    f"{filtered_df['traffic_volume'].min():.0f}"
                )

                st.subheader("Historical Data Preview")

                st.dataframe(
                    filtered_df[
                        ["date_time", "traffic_volume", "temp",
                         "rain_1h", "weather_main"]
                    ].head(20),
                    use_container_width=True
                )

    except FileNotFoundError:
        st.error("Cleaned traffic dataset not found.")

# Traffic Prediction page

elif page == "Traffic Prediction":

    st.header("Traffic Prediction")

    st.write(
        "Explore next-hour traffic predictions "
        "using our trained deep learning models."
    )

    model_choice = st.selectbox(
        "Choose Prediction Model",
        [
            "Original LSTM",
            "Multivariate LSTM",
            "Transformer"
        ]
    )

    if st.button("Generate Predictions"):

        with st.spinner("Loading model and generating predictions..."):

            (
                original_lstm,
                multivariate_lstm,
                transformer,
                uni_data,
                multi_data,
                uni_scaler,
                multi_scaler
            ) = load_models()

            if model_choice == "Original LSTM":

                model = original_lstm
                X_test = uni_data["X_test"]
                y_test = uni_data["y_test"]
                scaler = uni_scaler
                y_train_reference = uni_data["y_train"]

            elif model_choice == "Multivariate LSTM":

                model = multivariate_lstm
                X_test = multi_data["X_test"]
                y_test = multi_data["y_test"]
                scaler = multi_scaler
                y_train_reference = multi_data["y_train"]

            else:

                model = transformer
                X_test = multi_data["X_test"]
                y_test = multi_data["y_test"]
                scaler = multi_scaler
                y_train_reference = multi_data["y_train"]
                
            predictions = model.predict(
                X_test,
                verbose=0
            )

            actual = scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()

            predicted = scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()
            
            
# Calculate relative congestion thresholds from training data
            training_traffic = scaler.inverse_transform(
                y_train_reference.reshape(-1, 1)
            ).flatten()

            low_threshold, moderate_threshold, high_threshold = np.percentile(
                training_traffic,
                [25, 50, 75]
            )

# Classify predicted traffic volume
            def classify_congestion(volume):

                if volume <= low_threshold:
                    return "Low"

                elif volume <= moderate_threshold:
                    return "Moderate"

                elif volume <= high_threshold:
                    return "High"

                else:
                    return "Severe"

            congestion_levels = [
                classify_congestion(value)
                for value in predicted
            ]


            mae = mean_absolute_error(actual, predicted)

            rmse = np.sqrt(
                mean_squared_error(actual, predicted)
            )

            r2 = r2_score(actual, predicted)

            st.success("Predictions generated successfully!")

            # Display metrics
            col1, col2, col3 = st.columns(3)

            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("R² Score", f"{r2:.4f}")

            st.divider()

            # Display actual vs predicted graph
            st.subheader("Actual vs Predicted Traffic Volume")

            sample_count = min(300, len(actual))

            chart_data = pd.DataFrame({
                "Actual Traffic": actual[:sample_count],
                "Predicted Traffic": predicted[:sample_count]
            })

            st.line_chart(chart_data)

            # Display prediction table
            st.subheader("Prediction Details")

            result_data = pd.DataFrame({
                "Actual Traffic": actual[:20],
                "Predicted Traffic": predicted[:20],
                "Congestion Level": congestion_levels[:20]
            })

            result_data["Prediction Error"] = (
                result_data["Actual Traffic"]
                - result_data["Predicted Traffic"]
            )

            st.dataframe(
                result_data.round(2),
                use_container_width=True
            )

# Display congestion distribution
            st.subheader("Predicted Congestion Distribution")

            congestion_counts = pd.Series(
                congestion_levels
            ).value_counts().reindex(
                ["Low", "Moderate", "High", "Severe"],
                fill_value=0
            )

            st.bar_chart(congestion_counts)

            result_data["Prediction Error"] = (
                result_data["Actual Traffic"]
                - result_data["Predicted Traffic"]
            )

            st.dataframe(
                result_data.round(2),
                use_container_width=True
            )
            
            # Download prediction results
            report = pd.DataFrame({
                "Actual Traffic Volume": actual,
                "Predicted Traffic Volume": predicted,
                "Congestion Level": congestion_levels,
                "Prediction Error": actual - predicted
            })

            csv_data = report.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Prediction Report",
                data=csv_data,
                file_name="traffic_prediction_report.csv",
                mime="text/csv"
            )


    st.header("Traffic Prediction")

    st.write(
        "Select a trained model to explore its traffic predictions."
    )

    model_choice = st.selectbox(
        "Choose Model",
        ["Original LSTM", "Multivariate LSTM", "Transformer"]
    )

    st.write("Selected model:", model_choice)

    st.info(
        "We will connect the trained model and prediction interface in the next step."
    )

# Model Comparison page

elif page == "Model Comparison":

    st.header("Deep Learning Model Comparison")

    st.write(
        "Compare the performance of three deep learning "
        "models trained for next-hour traffic prediction."
    )

    # Load comparison results
    try:
        comparison = pd.read_csv(
            "models/three_model_comparison.csv"
        )

        st.success("Model comparison results loaded!")

        # Display comparison table
        st.subheader("Performance Metrics")

        st.dataframe(
            comparison.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Display individual metrics
        st.subheader("Metric Comparison")

        metric_choice = st.selectbox(
            "Select Evaluation Metric",
            ["MAE", "RMSE", "R2"]
        )

        chart_data = comparison.set_index("Model")[
            [metric_choice]
        ]

        st.bar_chart(chart_data)

        # Explain metrics
        explanations = {
            "MAE": "Average absolute difference between actual and predicted traffic volume. Lower is better.",
            "RMSE": "Penalizes larger prediction errors more heavily. Lower is better.",
            "R2": "Measures how much variation in traffic volume is explained by the model. Higher is better."
        }

        st.info(explanations[metric_choice])

        st.divider()

        # Display all three metric graphs
        st.subheader("Complete Model Performance")

        st.image(
            "models/three_model_metrics.png",
            caption="Comparison of MAE, RMSE and R² across all models",
            use_container_width=True
        )

        st.subheader("Actual vs Predicted Traffic")

        st.image(
            "models/three_model_predictions.png",
            caption="Actual traffic compared with all three model predictions",
            use_container_width=True
        )

    except FileNotFoundError:
        st.error(
            "Comparison files not found. "
            "Please run python src/compare_all_models.py first."
        )


    st.header("Model Performance Comparison")

    st.write("""
    Three deep learning architectures were trained and evaluated:
    - Original LSTM
    - Multivariate LSTM
    - Transformer
    """)

    st.info(
        "We will load the actual comparison results and graphs in the next step."
    )

# About Project page

elif page == "About Project":

    st.header("About the Project")

    st.markdown("""
    ## 🚦 Traffic Congestion Prediction System

    An AI-powered system that uses historical traffic and
    weather information to predict traffic volume for the
    next hour using deep learning.
    """)

    st.divider()

    # Project objective
    st.subheader("🎯 Project Objective")

    st.write("""
    The objective is to analyze historical traffic patterns
    and develop deep learning models capable of predicting
    future traffic volume.
    """)

    # Dataset information
    st.subheader("📊 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Records", "40,575")
        st.metric("Input Features", "24")

    with col2:
        st.metric("Dataset Period", "2012–2018")
        st.metric("Prediction Horizon", "1 Hour")

    st.info("""
    Dataset: Metro Interstate Traffic Volume.

    Location: Westbound I-94, Minneapolis–St. Paul, USA.

    This project currently uses historical data, not live
    traffic feeds.
    """)

    st.divider()

    # Models
    st.subheader("🧠 Deep Learning Models")

    st.markdown("""
    **1. Original LSTM**
    - Uses historical traffic volume.
    - Learns sequential traffic patterns.

    **2. Multivariate LSTM**
    - Uses traffic, weather and time-related features.
    - Learns relationships between multiple variables.

    **3. Transformer**
    - Uses positional encoding and self-attention.
    - Learns relationships across historical time steps.
    """)

    st.divider()

    # Technologies
    st.subheader("🛠️ Technologies Used")

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "TensorFlow / Keras",
        "LSTM Neural Networks",
        "Transformer Architecture",
        "Matplotlib",
        "Streamlit"
    ]

    st.write(" | ".join(technologies))

    st.divider()

    # Workflow
    st.subheader("⚙️ Project Workflow")

    st.markdown("""
    1. Historical traffic dataset collection.
    2. Data exploration and analysis.
    3. Data cleaning and preprocessing.
    4. Time-series sequence preparation.
    5. Deep learning model training.
    6. Model evaluation and comparison.
    7. Traffic prediction dashboard development.
    """)

    st.divider()

    st.caption(
        "Traffic Congestion Prediction System | "
        "AI & Data Science Project"
    )


    st.header("About the Project")

    st.write("""
    **Traffic Congestion Prediction System**

    This project uses historical traffic and weather information
    to predict next-hour traffic volume.

    Technologies used:

    - Python
    - Pandas and NumPy
    - TensorFlow / Keras
    - LSTM
    - Transformer
    - Scikit-learn
    - Streamlit
    """)
