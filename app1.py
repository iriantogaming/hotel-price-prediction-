
import streamlit as st
import pandas as pd
import joblib
from datetime import date

model = joblib.load("hotel_model1.pkl")
room_encoder = joblib.load("room_encoder1.pkl")

st.title("🏨 Hotel Room Price Prediction")

arrival_date = st.date_input("Arrival Date", value=date.today())

weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)

week_nights = st.number_input("Weekday Nights", min_value=0, value=2)

total_guests = st.number_input("Total Guests", min_value=1, value=2)

room_type = st.selectbox("Room Type", room_encoder.classes_)

if st.button("Predict Price"):

    arrival_year = arrival_date.year
    arrival_month_num = arrival_date.month
    arrival_day = arrival_date.day

    total_nights = weekend_nights + week_nights

    day_of_week = arrival_date.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0

    room_encoded = room_encoder.transform([room_type])[0]

    data = pd.DataFrame([[
        arrival_year,
        arrival_month_num,
        arrival_day,
        weekend_nights,
        week_nights,
        total_guests,
        room_encoded,
        total_nights,
        is_weekend
    ]], columns=[
        "arrival_date_year",
        "arrival_month_num",
        "arrival_date_day_of_month",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "total_guests",
        "reserved_room_type",
        "total_nights",
        "is_weekend"
    ])

    prediction = model.predict(data)[0]

    st.success(f"💰 Estimated Room Price: ${prediction:.2f} per night")

st.markdown("---")
st.subheader("📊 Model Performance Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="R-Squared ($R^2$)", value="0.4816")
with col2:
    st.metric(label="MAE", value="$24.9165")
with col3:
    st.metric(label="RMSE", value="$32.6299")
