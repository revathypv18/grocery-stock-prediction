import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Saved Model Components
# -----------------------------
model = joblib.load("rf_grocery_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")
feature_order = joblib.load("feature_order.pkl")

st.title("📦 Grocery Stock Status Prediction (Random Forest)")
st.write("Enter the details below to predict the stock availability.")

# -----------------------------
# Numeric Inputs
# -----------------------------
qty_bought = st.number_input("Quantity Bought", min_value=0, value=4)
days_used = st.number_input("Days Used", min_value=0, value=2)
daily_use = st.number_input("Daily Use", min_value=0, value=1)
last_purchase_days_ago = st.number_input("Last Purchase Days Ago", min_value=0, value=3)
household_size = st.number_input("Household Size", min_value=1, value=4)
price = st.number_input("Price", min_value=0.0, value=50.0)
discount_percent = st.number_input("Discount Percent", min_value=0.0, value=10.0)
shelf_life_days = st.number_input("Shelf Life Days", min_value=0, value=30)
rating = st.number_input("Rating", min_value=0, value=4)

# -----------------------------
# Categorical Inputs
# -----------------------------
item = st.number_input("Item (0–11)", min_value=0, max_value=11, value=0)
brand = st.number_input("Brand (0–4)", min_value=0, max_value=4, value=0)
store_type = st.number_input("Store Type (0–3)", min_value=0, max_value=3, value=0)
season = st.number_input("Season (0–3)", min_value=0, max_value=3, value=0)
purchase_method = st.number_input("Purchase Method (0–2)", min_value=0, max_value=2, value=0)

# -----------------------------
# Build Input DataFrame
# -----------------------------
input_df = pd.DataFrame({
    "item": [item],
    "brand": [brand],
    "store_type": [store_type],
    "season": [season],
    "purchase_method": [purchase_method],
    "qty_bought": [qty_bought],
    "days_used": [days_used],
    "daily_use": [daily_use],
    "last_purchase_days_ago": [last_purchase_days_ago],
    "household_size": [household_size],
    "price": [price],
    "discount_percent": [discount_percent],
    "shelf_life_days": [shelf_life_days],
    "rating": [rating]
})

# Ensure correct order
input_df = input_df[feature_order]

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Stock Status"):
    encoded_pred = model.predict(input_df)
    final_pred = target_encoder.inverse_transform(encoded_pred)

    st.success(f"📦 Predicted Stock Status: {final_pred[0]}")


    

