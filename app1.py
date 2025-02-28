import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import hashlib

# ✅ Persistent Storage for User Credentials
if "users_db" not in st.session_state:
    st.session_state["users_db"] = {}  # Store users persistently

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
# Store past transactions & fraud probabilities
if "transactions" not in st.session_state:
    st.session_state["transactions"] = []

def hash_password(password):
    """Hashes the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

# ✅ Signup Function
def signup():
    st.subheader("🔑 Create an Account")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")

    if st.button("Signup"):
        if new_user in st.session_state["users_db"]:
            st.error("❌ Username already exists. Try another one.")
        else:
            st.session_state["users_db"][new_user] = hash_password(new_pass)
            st.success("✅ Account created successfully! Please login.")

# ✅ Login Function
def login():
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        hashed_password = hash_password(password)
        if username in st.session_state["users_db"] and st.session_state["users_db"][username] == hashed_password:
            st.session_state["authenticated"] = True
            st.session_state["user"] = username
            st.sidebar.success(f"✅ Logged in as: {username}")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")

# ✅ Authentication Handling
if not st.session_state["authenticated"]:
    auth_choice = st.sidebar.radio("Select:", ["Login", "Signup"])
    if auth_choice == "Signup":
        signup()
    else:
        login()
    st.stop()  # Stop further execution until logged in

# ✅ Show Fraud Detection Options After Login
st.sidebar.success(f"👤 Logged in as: {st.session_state['user']}")

# 🔹 Load trained models
models = {
    "Credit Card Fraud": {
        "rf_model": joblib.load("Credit_Card_Fraud_rf_model.pkl"),
        "xgb_model": joblib.load("Credit_Card_Fraud_xgb_model.pkl"),
        "features": [
            "cc_num", "merchant", "category", "amount", "gender", "lat", "long", "city_pop",
            "trans_num", "unix_time", "day_of_week"
        ]
    },
    "UPI Fraud": {
        "rf_model": joblib.load("UPI_Fraud_rf_model.pkl"),
        "xgb_model": joblib.load("UPI_Fraud_xgb_model.pkl"),
        "features": [
            "amount", "MerchantCategory", "TransactionType", "Latitude", "Longitude", "AvgTransactionAmount",
            "TransactionFrequency", "UnusualLocation", "UnusualAmount", "NewDevice", "FailedAttempts",
            "hour", "day_of_week"
        ]
    },
    "Bank Account Fraud": {
        "rf_model": joblib.load("Bank_Account_Fraud_rf_model.pkl"),
        "xgb_model": joblib.load("Bank_Account_Fraud_xgb_model.pkl"),
        "features": [
            "income", "name_email_similarity", "prev_address_months_count", "current_address_months_count",
            "customer_age", "days_since_request", "amount", "payment_type", "zip_count_4w", "velocity_6h",
            "velocity_24h", "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
            "employment_status", "credit_risk_score", "email_is_free", "housing_status", "phone_home_valid",
            "phone_mobile_valid", "bank_months_count", "has_other_cards", "proposed_credit_limit",
            "foreign_request", "device_distinct_emails_8w", "device_fraud_count"
        ]
    }
}

# Sidebar Navigation - Show Only After Login
st.sidebar.title("🔍 Fraud Detection System")
menu = st.sidebar.radio("Choose Fraud Type:", ["Credit Card Fraud", "UPI Fraud", "Bank Account Fraud"])

# 🔹 Select the correct model & features
selected_model = models[menu]
features = selected_model["features"]

st.title(f"🚀 {menu} Detection")

# 🔹 User Inputs
user_inputs = {}

for feature in features:
    if feature in ["amount", "AvgTransactionAmount"]:
        user_inputs[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.1)
    elif "Latitude" in feature or "Longitude" in feature:
        user_inputs[feature] = st.number_input(f"{feature}", step=0.0001)
    elif "TransactionFrequency" in feature or "FailedAttempts" in feature or "velocity" in feature:
        user_inputs[feature] = st.number_input(f"{feature}", min_value=0, step=1)
    elif feature in ["category", "MerchantCategory", "TransactionType", "payment_type", "employment_status", "housing_status"]:
        user_inputs[feature] = st.text_input(f"{feature}", "Unknown")
    elif feature in ["gender", "email_is_free", "phone_home_valid", "phone_mobile_valid", "foreign_request", "has_other_cards"]:
        user_inputs[feature] = st.selectbox(f"{feature}", ["Yes", "No"])
    else:
        user_inputs[feature] = st.text_input(f"{feature}", "")

# 🔹 Predict Fraud on Button Click
def prepare_input_data(user_inputs, feature_list):
    df = pd.DataFrame([user_inputs], columns=feature_list)
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert all object columns to numeric
    df = df.fillna(0)  # Fill missing values with 0
    return df

def predict_fraud(input_data, rf_model, xgb_model, feature_list):
    input_df = prepare_input_data(input_data, feature_list)
    rf_pred = rf_model.predict_proba(input_df)[:, 1]
    xgb_pred = xgb_model.predict_proba(input_df)[:, 1]
    final_prob = (rf_pred + xgb_pred) / 2
    return final_prob[0]

if st.button("Predict Fraud"):
    input_data = list(user_inputs.values())

    fraud_prob = predict_fraud(input_data, selected_model["rf_model"], selected_model["xgb_model"], features)
    
    risk_level = "High Risk 🚨" if fraud_prob > 0.75 else "Medium Risk ⚠️" if fraud_prob > 0.5 else "Low Risk ✅"
    st.metric(label="Fraud Probability", value=f"{fraud_prob:.2%}")
    st.success(f"Risk Level: {risk_level}")

    # Store the transaction for visualization
    transaction_record = {"Transaction Amount": user_inputs.get("amount", 0), "Fraud Probability": fraud_prob}
    st.session_state["transactions"].append(transaction_record)

# 🔹 Insights & Graphs Based on User Inputs
st.subheader("📊 Fraud Trends & Insights")

# If transactions exist, visualize them
if len(st.session_state["transactions"]) > 0:
    df_transactions = pd.DataFrame(st.session_state["transactions"])

    # 🔹 Graph 1: Fraud Probability vs Transaction Amount
    fig1 = px.scatter(df_transactions, x="Transaction Amount", y="Fraud Probability", color="Fraud Probability",
                      title="Fraud Probability vs Transaction Amount")
    st.plotly_chart(fig1)

    # 🔹 Graph 2: Fraud Trends Over Time
    df_transactions["Index"] = range(1, len(df_transactions) + 1)
    fig2 = px.line(df_transactions, x="Index", y="Fraud Probability", title="Fraud Probability Trends Over Time")
    st.plotly_chart(fig2)
else:
    st.info("⚠️ No transactions yet. Make a prediction to see insights.")

st.sidebar.info("💡 Use the navigation to check fraud for different types of transactions.")