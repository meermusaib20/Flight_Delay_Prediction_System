# =========================================================
# Flight Delay Prediction Dashboard
# Author: Mir Musaib
# =========================================================

import streamlit as st
import pandas as pd
import pickle
import requests
import datetime
from geopy.distance import geodesic

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

model = pickle.load(open("flight_delay_model.pkl", "rb"))

# ---------------------------------------------------------
# MONTH MAP
# ---------------------------------------------------------

month_map = {
    "January":1,"February":2,"March":3,"April":4,
    "May":5,"June":6,"July":7,"August":8,
    "September":9,"October":10,"November":11,"December":12
}

# ---------------------------------------------------------
# AIRLINES
# ---------------------------------------------------------

airline_map = {
    "American Airlines":0,
    "Delta Airlines":1,
    "United Airlines":2,
    "Southwest Airlines":3,
    "JetBlue":4
}

# ---------------------------------------------------------
# AIRPORTS
# ---------------------------------------------------------

airport_map = {
    "JFK - New York":0,
    "LAX - Los Angeles":1,
    "ORD - Chicago":2,
    "ATL - Atlanta":3,
    "DFW - Dallas":4
}

# ---------------------------------------------------------
# AIRPORT COORDINATES
# ---------------------------------------------------------

airport_coordinates = {
    "JFK - New York":(40.6413,-73.7781),
    "LAX - Los Angeles":(33.9416,-118.4085),
    "ORD - Chicago":(41.9742,-87.9073),
    "ATL - Atlanta":(33.6407,-84.4277),
    "DFW - Dallas":(32.8998,-97.0403)
}

# ---------------------------------------------------------
# WEATHER API
# ---------------------------------------------------------

def get_weather_data(lat, lon):

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

    response = requests.get(url)

    data = response.json()

    temperature = data["current_weather"]["temperature"]
    windspeed = data["current_weather"]["windspeed"]

    return temperature, windspeed


# ---------------------------------------------------------
# ROUTE CONGESTION FUNCTION
# ---------------------------------------------------------

def get_route_congestion(origin, dest):

    lat1, lon1 = airport_coordinates[origin]
    lat2, lon2 = airport_coordinates[dest]

    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2

    url = "https://opensky-network.org/api/states/all"

    try:

        response = requests.get(url)

        data = response.json()

        flights = data["states"]

        count = 0

        for f in flights:

            if f[6] is not None and f[5] is not None:

                lat = f[6]
                lon = f[5]

                if abs(lat - mid_lat) < 5 and abs(lon - mid_lon) < 5:
                    count += 1

        return count

    except:
        return 0


# ---------------------------------------------------------
# DISTANCE CALCULATION
# ---------------------------------------------------------

def calculate_distance(origin, destination):

    coord1 = airport_coordinates[origin]
    coord2 = airport_coordinates[destination]

    distance_km = geodesic(coord1, coord2).km

    return distance_km


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("✈ Flight Delay Prediction Dashboard")

st.write(
"""
Predict whether a flight will be **delayed or on time** using machine learning  
combined with **real-time weather and air traffic congestion data**.
"""
)

st.divider()

# ---------------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------------

st.sidebar.header("Flight Details")

month_name = st.sidebar.selectbox("Month", list(month_map.keys()))
Month = month_map[month_name]

flight_date = st.sidebar.date_input("Flight Date")

DayOfWeek = flight_date.weekday() + 1

DepHour = st.sidebar.slider("Departure Hour",0,23)
ArrHour = st.sidebar.slider("Arrival Hour",0,23)

airline_name = st.sidebar.selectbox("Airline", list(airline_map.keys()))
UniqueCarrier = airline_map[airline_name]

origin_airport = st.sidebar.selectbox("Origin Airport", list(airport_map.keys()))
Origin = airport_map[origin_airport]

dest_airport = st.sidebar.selectbox("Destination Airport", list(airport_map.keys()))
Dest = airport_map[dest_airport]

# ---------------------------------------------------------
# DISTANCE
# ---------------------------------------------------------

Distance = calculate_distance(origin_airport, dest_airport)

st.subheader("Flight Route")

st.metric("Estimated Distance (km)", int(Distance))

# ---------------------------------------------------------
# WEATHER
# ---------------------------------------------------------

lat, lon = airport_coordinates[origin_airport]

temperature, windspeed = get_weather_data(lat, lon)

st.subheader("🌦 Weather at Origin Airport")

col1, col2 = st.columns(2)

col1.metric("Temperature °C", temperature)
col2.metric("Wind Speed km/h", windspeed)

# ---------------------------------------------------------
# CONGESTION
# ---------------------------------------------------------

traffic = get_route_congestion(origin_airport, dest_airport)

st.subheader("✈ Route Air Traffic")

st.metric("Flights along route", traffic)

# ---------------------------------------------------------
# FEATURE ENGINEERING
# (scaled so model doesn't always predict delay)
# ---------------------------------------------------------

WeatherDelay = windspeed * 0.02

NASDelay = traffic / 200

LateAircraftDelay = 0

# ---------------------------------------------------------
# MODEL INPUT
# ---------------------------------------------------------

input_data = pd.DataFrame({
    "Month":[Month],
    "DayOfWeek":[DayOfWeek],
    "DepHour":[DepHour],
    "ArrHour":[ArrHour],
    "UniqueCarrier":[UniqueCarrier],
    "Origin":[Origin],
    "Dest":[Dest],
    "Distance":[Distance],
    "WeatherDelay":[WeatherDelay],
    "NASDelay":[NASDelay],
    "LateAircraftDelay":[LateAircraftDelay]
})

# ---------------------------------------------------------
# OPTIONAL DEBUG
# ---------------------------------------------------------

# st.write(input_data)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

st.divider()

if st.button("🚀 Predict Flight Delay"):

    prob = model.predict_proba(input_data)[0][1]

    if prob > 0.6:
        st.error("⚠ Flight likely to be delayed")
    else:
        st.success("✅ Flight likely on time")

    st.subheader("Delay Probability")

    st.progress(int(prob * 100))

    st.write(f"Probability of Delay: **{prob:.2%}**")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption("Flight Delay Prediction System | Mir Musaib | Data Science Internship")