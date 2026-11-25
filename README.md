# IntelliRoute AI — Traffic Congestion Prediction

An academic prototype combining interactive route alternatives with deep-learning traffic forecasting.

## Features

- Enter an origin and destination and view road routes on an interactive map.
- Compare alternative routes by baseline duration and distance, with a transparent relative score and adjustable time-versus-distance preference.
- Explore a historical day-of-week versus hour heatmap for the I-94 traffic sensor.
- Compare the three model outputs, inspect their disagreement, and view recent observations alongside the next-hour forecast.
- Select a historical observation and forecast the following hour using three trained models: Original LSTM, Multivariate LSTM, and Transformer.
- Display model outputs separately and explain the data limitations.

## Dataset and model scope

The models were trained on the Metro Interstate Traffic Volume dataset: a historical westbound I-94 traffic sensor series in the Minneapolis–St. Paul area (2012–2018). The models do **not** predict congestion for every road segment or for arbitrary cities.

The route planner uses OpenStreetMap Nominatim for place lookup and the public OSRM demo routing service for road geometry and baseline route duration. These services do not provide live traffic in this application. Route estimates are therefore not live-traffic ETAs. The historical AI forecast is deliberately kept separate from route scoring so the app does not imply unsupported segment-level predictions.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Ensure the trained model files and processed data files are present locally:

- `models/traffic_lstm.keras`
- `models/traffic_multivariate_lstm.keras`
- `models/traffic_transformer.keras`
- `models/traffic_scaler.pkl`
- `models/multivariate_feature_scaler.pkl`
- `models/multivariate_target_scaler.pkl`
- `data/processed/traffic_cleaned.csv`

The generated `data/processed/multivariate_lstm_data.npz` is not required by the route planner page; the saved models and scalers are used directly. If it is excluded from Git because of file size, regenerate it with:

```powershell
python src/prepare_multivariate_data.py
```

## External services

- OpenStreetMap Nominatim geocoding: follow the usage policy; requests are cached and use an identifying User-Agent.
- OSRM public demo server: suitable for prototypes and demonstrations, not production-scale use. For deployment, configure a hosted routing provider or self-host OSRM.

## Important limitations

- No live traffic feed is connected.
- The historical heatmap is temporal analysis of one I-94 sensor, not a geographic congestion layer for all roads.
- Route scores compare OSRM baseline duration and distance only; they are not AI-learned or live-traffic congestion scores.
- Model disagreement is a simple diagnostic, not a calibrated confidence interval.
- Route durations are baseline estimates from the routing service.
- AI forecasts are historical I-94 corridor demonstrations, not current conditions.
- Congestion levels based on model outputs should not be interpreted as official road-capacity classifications.

## Technology

Python, Streamlit, TensorFlow/Keras, Pandas, NumPy, scikit-learn, Folium, Streamlit-Folium, OpenStreetMap, OSRM.
