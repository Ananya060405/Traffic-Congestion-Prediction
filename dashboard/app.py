from pathlib import Path
from datetime import date
import requests
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "traffic_cleaned.csv"
MODEL_DIR = ROOT / "models"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
OSRM = "https://router.project-osrm.org/route/v1/driving"

st.set_page_config(page_title="IntelliRoute AI", page_icon="🚦", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {padding: 1.25rem 1.5rem; border-radius: 14px; background: linear-gradient(115deg,#10243a,#164e63); color: white; margin-bottom: 1rem;}
.hero h1 {font-size: 2rem; margin: 0;}
.hero p {margin: .4rem 0 0; color: #dbeafe;}
.small-note {font-size: .85rem; color: #64748b;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_traffic_data():
    df = pd.read_csv(DATA_FILE)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)
    return df

@st.cache_resource
def load_models():
    return {
        "lstm": tf.keras.models.load_model(MODEL_DIR / "traffic_lstm.keras"),
        "multi": tf.keras.models.load_model(MODEL_DIR / "traffic_multivariate_lstm.keras"),
        "transformer": tf.keras.models.load_model(MODEL_DIR / "traffic_transformer.keras"),
        "uni_scaler": joblib.load(MODEL_DIR / "traffic_scaler.pkl"),
        "feature_scaler": joblib.load(MODEL_DIR / "multivariate_feature_scaler.pkl"),
        "target_scaler": joblib.load(MODEL_DIR / "multivariate_target_scaler.pkl"),
    }

@st.cache_data(ttl=3600)
def geocode(place):
    response = requests.get(
        NOMINATIM,
        params={"q": place, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": "IntelliRouteAI/1.0 educational-project"},
        timeout=20,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        return None
    item = results[0]
    return {"name": item["display_name"], "lat": float(item["lat"]), "lon": float(item["lon"])}

@st.cache_data(ttl=300)
def get_routes(start_lon, start_lat, end_lon, end_lat):
    coordinates = f"{start_lon},{start_lat};{end_lon},{end_lat}"
    response = requests.get(
        f"{OSRM}/{coordinates}",
        params={"alternatives": "true", "steps": "true", "overview": "full", "geometries": "geojson"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != "Ok" or not payload.get("routes"):
        raise ValueError("The routing service could not find a drivable route.")
    return payload["routes"]

def prepare_multivariate_features(df, feature_columns):
    work = df.copy()
    work["is_holiday"] = (work["holiday"].fillna("No Holiday") != "No Holiday").astype(int)
    work["hour_sin"] = np.sin(2 * np.pi * work["date_time"].dt.hour / 24)
    work["hour_cos"] = np.cos(2 * np.pi * work["date_time"].dt.hour / 24)
    work["day_sin"] = np.sin(2 * np.pi * work["date_time"].dt.dayofweek / 7)
    work["day_cos"] = np.cos(2 * np.pi * work["date_time"].dt.dayofweek / 7)
    work["month_sin"] = np.sin(2 * np.pi * (work["date_time"].dt.month - 1) / 12)
    work["month_cos"] = np.cos(2 * np.pi * (work["date_time"].dt.month - 1) / 12)
    weather = pd.get_dummies(work["weather_main"], prefix="weather", dtype=int)
    work = pd.concat([work, weather], axis=1)
    for column in feature_columns:
        if column not in work.columns:
            work[column] = 0
    return work[feature_columns].astype(float)

def predict_historical_snapshot(df, selected_time, models):
    positions = df.index[df["date_time"] == pd.Timestamp(selected_time)].tolist()
    if not positions:
        raise ValueError("No record exists for the selected date and time.")
    end = positions[0] + 1
    start = end - 24
    if start < 0:
        raise ValueError("There are not 24 earlier observations for this time.")
    window = df.iloc[start:end].copy()
    if not (window["date_time"].diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("The previous 24 records are not continuous hourly observations. Choose another time.")

    uni_values = window["traffic_volume"].to_numpy(dtype=float).reshape(-1, 1)
    uni_scaled = models["uni_scaler"].transform(uni_values).reshape(1, 24, 1)
    lstm_scaled = models["lstm"].predict(uni_scaled, verbose=0).reshape(-1, 1)
    lstm_value = float(models["uni_scaler"].inverse_transform(lstm_scaled)[0, 0])

    feature_columns = list(models["feature_scaler"].feature_names_in_)
    all_features = prepare_multivariate_features(df, feature_columns)
    multi_window = all_features.iloc[start:end].to_numpy(dtype=float)
    multi_scaled = models["feature_scaler"].transform(multi_window).reshape(1, 24, len(feature_columns))
    multi_model_value = float(models["target_scaler"].inverse_transform(
        models["multi"].predict(multi_scaled, verbose=0).reshape(-1, 1)
    )[0, 0])
    transformer_value = float(models["target_scaler"].inverse_transform(
        models["transformer"].predict(multi_scaled, verbose=0).reshape(-1, 1)
    )[0, 0])
    return {
        "Original LSTM": max(0, lstm_value),
        "Multivariate LSTM": max(0, multi_model_value),
        "Transformer": max(0, transformer_value),
        "observed": float(window["traffic_volume"].iloc[-1]),
        "forecast_time": pd.Timestamp(selected_time) + pd.Timedelta(hours=1),
    }

def congestion_label(value, thresholds):
    q1, q2, q3 = thresholds
    if value <= q1:
        return "Low"
    if value <= q2:
        return "Moderate"
    if value <= q3:
        return "High"
    return "Severe"

st.markdown("""
<div class="hero">
  <h1>🚦 IntelliRoute AI</h1>
  <p>Interactive route planning, alternative-route comparison and historical AI traffic forecasting</p>
</div>
""", unsafe_allow_html=True)

route_tab, forecast_tab, insights_tab, about_tab = st.tabs(["🗺️ Smart Route Planner", "🧠 AI Traffic Forecast", "📊 Historical Insights", "ℹ️ About"])

with route_tab:
    st.subheader("Plan a journey")
    left, right = st.columns(2)
    with left:
        origin_text = st.text_input("Starting point", placeholder="e.g. Mall of America, Bloomington, Minnesota")
    with right:
        destination_text = st.text_input("Destination", placeholder="e.g. Target Field, Minneapolis, Minnesota")
    optimize = st.radio("Recommendation strategy", ["Balanced recommendation", "Fastest journey", "Shortest distance"], horizontal=True)
    time_priority = st.slider("Travel-time priority for balanced scoring", 0, 100, 70, 5, help="The remaining weight is assigned to shorter distance. This compares routing-service estimates, not live congestion.")
    plan = st.button("Find routes", type="primary", use_container_width=False)

    if plan:
        if not origin_text.strip() or not destination_text.strip():
            st.warning("Enter both a starting point and a destination.")
        else:
            try:
                with st.spinner("Finding locations and calculating alternative routes..."):
                    origin = geocode(origin_text.strip())
                    destination = geocode(destination_text.strip())
                    if origin is None or destination is None:
                        raise ValueError("Could not locate one of the places. Try adding the city and state.")
                    routes = get_routes(origin["lon"], origin["lat"], destination["lon"], destination["lat"])
                st.session_state["route_result"] = {
                    "origin": origin, "destination": destination, "routes": routes,
                    "optimize": optimize, "time_priority": time_priority,
                }
            except Exception as exc:
                st.error(f"Unable to calculate routes: {exc}")

    result = st.session_state.get("route_result")
    if result:
        routes = result["routes"]
        durations = np.array([route["duration"] for route in routes], dtype=float)
        distances = np.array([route["distance"] for route in routes], dtype=float)

        def normalized(values):
            span = float(values.max() - values.min())
            return np.zeros_like(values) if span == 0 else (values - values.min()) / span

        if result["optimize"] == "Fastest journey":
            route_costs = normalized(durations)
            score_labels = "fastest"
        elif result["optimize"] == "Shortest distance":
            route_costs = normalized(distances)
            score_labels = "shortest"
        else:
            weight_time = result["time_priority"] / 100
            route_costs = weight_time * normalized(durations) + (1 - weight_time) * normalized(distances)
            score_labels = "balanced"
        route_scores = 100 * (1 - route_costs)
        best_index = int(np.argmax(route_scores))
        colors = ["#087f5b", "#2563eb", "#d97706", "#7c3aed"]
        center = [
            (result["origin"]["lat"] + result["destination"]["lat"]) / 2,
            (result["origin"]["lon"] + result["destination"]["lon"]) / 2,
        ]
        route_map = folium.Map(location=center, zoom_start=11, control_scale=True)
        folium.Marker(
            [result["origin"]["lat"], result["origin"]["lon"]],
            tooltip="Start: " + result["origin"]["name"],
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(route_map)
        folium.Marker(
            [result["destination"]["lat"], result["destination"]["lon"]],
            tooltip="Destination: " + result["destination"]["name"],
            icon=folium.Icon(color="red", icon="flag"),
        ).add_to(route_map)
        for i, route in enumerate(routes):
            coords = route["geometry"]["coordinates"]
            latlon = [[point[1], point[0]] for point in coords]
            folium.PolyLine(
                latlon, color=colors[i % len(colors)],
                weight=7 if i == best_index else 4,
                opacity=0.9 if i == best_index else 0.65,
                tooltip=f"Route {i + 1}" + (" — recommended" if i == best_index else ""),
            ).add_to(route_map)
        st_folium(route_map, use_container_width=True, height=520, returned_objects=[])

        st.subheader("Compare your route options")
        st.caption("The routing service supplies road distance and baseline duration. These are not live-traffic ETAs.")
        cards = st.columns(min(len(routes), 3))
        for i, route in enumerate(routes):
            with cards[i % len(cards)]:
                st.markdown(f"**Route {i + 1}**")
                if i == best_index:
                    st.success("Recommended")
                st.metric("Estimated driving time", f"{route['duration'] / 60:.0f} min")
                st.metric("Distance", f"{route['distance'] / 1000:.1f} km")
                st.metric("Relative route score", f"{route_scores[i]:.0f}/100")
        st.info(
            f"Recommended: Route {best_index + 1}. It has the strongest relative {score_labels} score among the alternatives returned. "
            "Scores compare OSRM baseline duration and distance only; they are not AI-predicted or live-traffic congestion scores."
        )
        st.caption("Map data © OpenStreetMap contributors. Route geometry and baseline estimates provided by the OSRM demo service.")

with forecast_tab:
    st.subheader("Forecast traffic on the I-94 training corridor")
    st.write("Choose a historical observation from the Metro Interstate Traffic Volume dataset. The trained models use the previous 24 continuous hourly records to forecast the following hour.")
    if not DATA_FILE.exists():
        st.error("The processed traffic dataset is missing. Run the data preparation scripts first.")
    else:
        df = load_traffic_data()
        min_day = df["date_time"].min().date()
        max_day = df["date_time"].max().date()
        chosen_day = st.date_input("Historical reference date", value=max_day, min_value=min_day, max_value=max_day)
        day_rows = df[df["date_time"].dt.date == chosen_day]
        if day_rows.empty:
            st.warning("No observations exist for that date.")
        else:
            options = day_rows["date_time"].dt.strftime("%H:%M").tolist()
            chosen_clock = st.selectbox("Last observed time", options, index=len(options) - 1)
            selected_time = pd.Timestamp(f"{chosen_day} {chosen_clock}")
            if st.button("Run all three AI models", type="primary"):
                try:
                    with st.spinner("Loading trained models and forecasting..."):
                        models = load_models()
                        forecasts = predict_historical_snapshot(df, selected_time, models)
                    st.session_state["forecast_result"] = forecasts
                except Exception as exc:
                    st.error(f"Forecast could not be generated: {exc}")

        forecast = st.session_state.get("forecast_result")
        if forecast:
            st.caption(f"Forecast target: {forecast['forecast_time']:%d %b %Y, %H:%M} (historical dataset time)")
            a, b, c = st.columns(3)
            for col, model_name in zip([a, b, c], ["Original LSTM", "Multivariate LSTM", "Transformer"]):
                with col:
                    st.metric(model_name, f"{forecast[model_name]:,.0f} vehicles/hour")
            avg = float(np.mean([forecast["Original LSTM"], forecast["Multivariate LSTM"], forecast["Transformer"]]))
            st.metric("Ensemble mean (informational)", f"{avg:,.0f} vehicles/hour")
            st.metric("Last observed traffic", f"{forecast['observed']:,.0f} vehicles/hour")
            predictions = [forecast["Original LSTM"], forecast["Multivariate LSTM"], forecast["Transformer"]]
            model_spread = max(predictions) - min(predictions)
            mean_prediction = float(np.mean(predictions))
            spread_pct = (model_spread / mean_prediction * 100) if mean_prediction else 0.0
            st.metric("Model disagreement (max − min)", f"{model_spread:,.0f} vehicles/hour", f"{spread_pct:.1f}% of ensemble mean")
            st.caption("Model disagreement is a comparison between model outputs, not a calibrated confidence interval.")

            reference_time = forecast["forecast_time"] - pd.Timedelta(hours=1)
            history_end = df.index[df["date_time"] == reference_time].tolist()
            if history_end:
                recent = df.iloc[max(0, history_end[0] - 23):history_end[0] + 1][["date_time", "traffic_volume"]]
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.plot(recent["date_time"], recent["traffic_volume"], marker="o", label="Observed history")
                ax.scatter([forecast["forecast_time"]], [mean_prediction], marker="*", s=180, label="Three-model mean forecast")
                ax.set_title("Previous observations and next-hour model forecast")
                ax.set_ylabel("Vehicles per hour")
                ax.set_xlabel("Historical dataset time")
                ax.grid(alpha=0.25)
                ax.legend()
                fig.autofmt_xdate()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.info("These forecasts are historical I-94 model demonstrations, not current traffic conditions and not predictions for the route selected in the map.")
            with st.expander("What makes the models different?"):
                st.write("- Original LSTM: uses the previous 24 traffic-volume observations.")
                st.write("- Multivariate LSTM: uses traffic, weather and time-related features.")
                st.write("- Transformer: uses the same multivariate inputs with attention layers.")

with insights_tab:
    st.subheader("Historical I-94 congestion patterns")
    st.write("Explore how recorded traffic volume varied by hour and day of week in the training dataset. Categories are based on this dataset's own traffic-volume quartiles; these are historical patterns, not live road conditions.")
    if DATA_FILE.exists():
        insight_df = load_traffic_data().copy()
        insight_df["day_name"] = insight_df["date_time"].dt.day_name()
        insight_df["hour_of_day"] = insight_df["date_time"].dt.hour
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat = insight_df.pivot_table(index="day_name", columns="hour_of_day", values="traffic_volume", aggfunc="mean").reindex(day_order)
        q1, q2, q3 = insight_df["traffic_volume"].quantile([0.25, 0.50, 0.75]).tolist()
        fig, ax = plt.subplots(figsize=(12, 4.5))
        image = ax.imshow(heat.to_numpy(), aspect="auto", interpolation="nearest", cmap="YlOrRd")
        ax.set_yticks(range(len(heat.index)), labels=[d[:3] for d in heat.index])
        ax.set_xticks(range(len(heat.columns)), labels=[str(int(h)) for h in heat.columns])
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Day of week")
        ax.set_title("Average historical traffic volume (vehicles/hour)")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Vehicles per hour")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        peak = heat.stack().idxmax()
        peak_value = float(heat.stack().max())
        avg_by_hour = insight_df.groupby("hour_of_day")["traffic_volume"].mean()
        peak_hour = int(avg_by_hour.idxmax())
        a, b, c = st.columns(3)
        a.metric("Busiest average hour", f"{peak_hour:02d}:00")
        b.metric("Busiest day/hour cell", f"{peak[0]}, {int(peak[1]):02d}:00")
        c.metric("Peak cell average", f"{peak_value:,.0f} vehicles/hour")
        st.caption(f"Dataset-specific traffic-volume bands: Low ≤ {q1:,.0f}; Moderate ≤ {q2:,.0f}; High ≤ {q3:,.0f}; Severe > {q3:,.0f} vehicles/hour.")
        st.markdown("#### Average traffic by hour")
        st.line_chart(avg_by_hour)
        st.warning("This is a temporal heatmap for the single westbound I-94 sensor. It cannot be overlaid as a geographic congestion heatmap on arbitrary routes without road-segment traffic measurements.")
    else:
        st.error("The processed traffic dataset is missing.")

with about_tab:
    st.subheader("Project scope and data transparency")
    st.markdown("""
    **IntelliRoute AI** combines two working capabilities:

    1. **Route planning:** geocodes places, draws road routes and compares available alternatives using OpenStreetMap-based services.
    2. **Deep-learning forecast:** runs the three trained models on a selected historical I-94 observation window.

    **Training data:** Metro Interstate Traffic Volume, westbound I-94 in the Minneapolis–St. Paul area, 2012–2018.

    **Important limitation:** the training data represents one traffic sensor/time series. It does not contain road-segment IDs or traffic measurements for every road. Therefore the AI forecast is shown separately and is not falsely presented as live, route-specific congestion. Route durations are baseline routing estimates from OSRM.
    """)
    st.markdown("**Models:** Original LSTM · Multivariate LSTM · Transformer")
    st.markdown("**Stack:** Python · Streamlit · TensorFlow/Keras · Pandas · Folium · OpenStreetMap · OSRM")
    st.caption("Academic prototype — not a live navigation or emergency-routing service.")
