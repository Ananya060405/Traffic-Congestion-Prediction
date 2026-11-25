from pathlib import Path
from datetime import date
import requests
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
import folium
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

route_tab, forecast_tab, about_tab = st.tabs(["🗺️ Smart Route Planner", "🧠 AI Traffic Forecast", "ℹ️ About"])

with route_tab:
    st.subheader("Plan a journey")
    left, right = st.columns(2)
    with left:
        origin_text = st.text_input("Starting point", placeholder="e.g. Mall of America, Bloomington, Minnesota")
    with right:
        destination_text = st.text_input("Destination", placeholder="e.g. Target Field, Minneapolis, Minnesota")
    optimize = st.radio("Recommend route based on", ["Fastest journey", "Shortest distance"], horizontal=True)
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
                    "optimize": optimize,
                }
            except Exception as exc:
                st.error(f"Unable to calculate routes: {exc}")

    result = st.session_state.get("route_result")
    if result:
        routes = result["routes"]
        best_index = min(
            range(len(routes)),
            key=lambda i: routes[i]["duration"] if result["optimize"] == "Fastest journey" else routes[i]["distance"],
        )
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
        st.info(
            f"Recommended: Route {best_index + 1}, selected for the {result['optimize'].lower()}. "
            "The current model is trained on one historical I-94 traffic sensor and does not score congestion separately on each road segment."
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
            st.info("These forecasts are historical I-94 model demonstrations, not current traffic conditions and not predictions for the route selected in the map.")
            with st.expander("What makes the models different?"):
                st.write("- Original LSTM: uses the previous 24 traffic-volume observations.")
                st.write("- Multivariate LSTM: uses traffic, weather and time-related features.")
                st.write("- Transformer: uses the same multivariate inputs with attention layers.")

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
