
import streamlit as st
import pandas as pd
import joblib
from datetime import date, timedelta

model = joblib.load("hotel_model1.pkl")
room_encoder = joblib.load("room_encoder1.pkl")

st.title("🏨 Hotel Room Price Prediction")

# ── Input User ───────────────────────────────
arrival_date = st.date_input("Arrival Date", value=date.today())

checkout_date = st.date_input(
    "Checkout Date",
    value=date.today() + timedelta(days=3)
)

total_guests = st.number_input("Total Guests", min_value=1, value=2)

room_type = st.selectbox("Room Type", room_encoder.classes_)

# ── Kalkulasi Otomatis ───────────────────────
if checkout_date <= arrival_date:
    st.error("Checkout date harus setelah arrival date!")
else:
    total_nights = (checkout_date - arrival_date).days

    weekend_nights = 0
    week_nights = 0
    current = arrival_date
    while current < checkout_date:
        if current.weekday() >= 5:
            weekend_nights += 1
        else:
            week_nights += 1
        current += timedelta(days=1)

    st.info(f"📅 Total: {total_nights} malam ({weekend_nights} weekend, {week_nights} weekday)")

    if st.button("Predict Price"):
        arrival_year = arrival_date.year
        arrival_month_num = arrival_date.month
        arrival_day = arrival_date.day
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

        price_per_night = model.predict(data)[0]
        total_price = price_per_night * total_nights

        st.success(f"💰 Estimated Price: ${price_per_night:.2f} per night")
        st.success(f"🧾 Total Estimated Price: ${total_price:.2f} for {total_nights} nights")

# ── Metrics ──────────────────────────────────
st.markdown("---")
st.subheader("📊 Model Performance")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("R²", "0.4816")
with col2:
    st.metric("MAE", "$24.91")
with col3:
    st.metric("RMSE", "$32.62")
