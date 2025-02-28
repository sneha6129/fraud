import streamlit as st
from supabase import Client, create_client
import bcrypt

import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import joblib
import random
from datetime import datetime


supabase_url = "https://jzydlydujbhwdxqntsxu.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp6eWRseWR1amJod2R4cW50c3h1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA1NDc1MjMsImV4cCI6MjA1NjEyMzUyM30.K2j-XqM6wd56d53LRtBc_h3-pB_6sdX1fymxkLGlcIk"
supabase = create_client(supabase_url, supabase_key)

# Initialize session state
if "users_db" not in st.session_state:
    st.session_state["users_db"] = {}

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "transactions" not in st.session_state:
    try:
        transactions = supabase.table("Transactions").select("*").execute()
        st.session_state["transactions"] = pd.DataFrame(transactions.data)
    except Exception as e:
        st.session_state["transactions"] = pd.DataFrame(columns=[
            'Timestamp', 'Transaction_Type', 'Amount', 'Fraud_Probability', 
            'Risk_Level', 'Model_Used'
        ])
        st.error(f"Error fetching transactions: {e}")

st.title("Fraud Detection in Banking Data ")

if "reset_codes" not in st.session_state:
    st.session_state["reset_codes"] = {}

# Authentication Functions
def hash_password(password):
    # Generate a salt and hash the password with the salt
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')  # Return hashed password as string

# Function to verify the password using bcrypt
def verify_password(stored_hash, entered_password):
    # Compare the entered password with the stored hash
    return bcrypt.checkpw(entered_password.encode('utf-8'), stored_hash.encode('utf-8'))

def is_valid_password(password):
    # Password must have at least 6 characters, including uppercase, lowercase, special characters, and numbers
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{6,}$"
    return re.match(pattern, password)

def is_valid_username(username):
    # Username must be between 3 and 20 characters long and contain only alphanumeric characters
    pattern = r"^[a-zA-Z0-9_]{3,20}$"
    return re.match(pattern, username)

def signup():
    st.subheader("🔑 Create an Account")
    image_path = "login.jpg"
    st.sidebar.image(image_path, use_container_width=True)
    
    # User input fields
    new_user = st.text_input("Email")
    username = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    confirm_pass = st.text_input("Confirm Password", type="password")
    
    # Email pattern for validation
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    # Username pattern for validation
    if st.button("Signup"):
        if not re.match(email_pattern, new_user):
            st.error("❌ Invalid email format. Please enter a valid email address.")
        elif not is_valid_username(username):
            st.error("❌ Username must be between 3 and 20 characters and can only contain letters, numbers, and underscores.")
        elif not is_valid_password(new_pass):
            st.error("❌ Password must have at least 6 characters, including uppercase, lowercase, special characters, and numbers.")
        elif new_pass != confirm_pass:
            st.error("❌ Passwords do not match. Please try again.")
        else:
            try:
                # Insert new user into Supabase with bcrypt hashed password
                hashed_password = hash_password(new_pass)
                supabase.table("users").insert({
                    "email": new_user,
                    "username": username,
                    "password_hash": hashed_password
                }).execute()
                st.success("✅ Account created successfully! Please login.")
            except Exception as e:
                st.error(f"❌ Error creating account: {e}")

def forgot_password():
    st.subheader("🔒 Forgot Password")
    image_path = "login.jpg"
    st.sidebar.image(image_path, use_container_width=True)
    email = st.text_input("Enter your Email")

    try:
        response = supabase.from_("users").select("*").eq("email", email).execute()

        if response.data and len(response.data) > 0:
            user = response.data[0]  # Get the first matching user

            new_pass = st.text_input("Enter new password", type="password")
            confirm_pass = st.text_input("Confirm new password", type="password")
            
            if st.button("Reset Password"):
                if not is_valid_password(new_pass):
                    st.error("❌ Password must have at least 6 characters, including uppercase, lowercase, special characters, and numbers.")
                elif new_pass != confirm_pass:
                    st.error("❌ Passwords do not match. Please try again.")
                else:
                    supabase.from_("users").update({"password_hash": hash_password(new_pass)}).eq("email", email).execute()
                    st.success("✅ Password reset successfully! Please login.")
                    st.rerun()
        else:
            st.error("❌ Email not found.")
    except Exception as e:
        st.error(f"Error: {e}")




def login():
    st.subheader("🔐 Login")
    image_path = "login.jpg"
    st.sidebar.image(image_path, use_container_width=True)
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        try:
            # Fetch user from Supabase
            user = supabase.table("users").select("*").eq("email", email).execute()
            if user.data:
                stored_hash = user.data[0]["password_hash"]
                if verify_password(stored_hash, password):
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = email
                    st.sidebar.success(f"✅ Logged in as: {email}")
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password.")
            else:
                st.error("❌ User not found.")
        except Exception as e:
            st.error(f"❌ Error logging in: {e}")

def logout():
    st.session_state["authenticated"] = False
    st.session_state.pop("user", None)
    image_path = "login.jpg"
    st.sidebar.image(image_path, use_container_width=True)
    st.success("✅ Logged out successfully!")
    st.rerun()

# Load Models
models = {
    "Credit Card Fraud": {
        "rf_model": joblib.load("Credit_Card_Fraud_rf_model.pkl"),
        "xgb_model": joblib.load("Credit_Card_Fraud_xgb_model.pkl"),
        "features": [
            "cc_num", "merchant", "category", "amount", "gender", "lat", "long", "city_pop",
            "trans_num", "unix_time", "day_of_week"
        ],
        "numeric_ranges": {
            "amount": (1.0, 1e6),  # Adjusted max value
              # Adjusted max value
            "city_pop": (0, 10000000),  # Adjusted max value
            "trans_num": (0, 1000000),
            "unix_time": (0, 2**32 - 1),
            "day_of_week": (0, 6),
            "avg_trans_amount": (1.0, 1e6),  # Adjusted max value
            "failed_attempts": (0, 10),
            "trans_frequency": (0, 100),
            "hour": (0, 23),
             
            "lat":(-90.0,90.0),
            "long":(-180.0,180.0)
        },
        "categorical_features": ["category", "gender","merchant"]
    },
    "UPI Fraud": {
        "rf_model": joblib.load("UPI_Fraud_rf_model.pkl"),
        "xgb_model": joblib.load("UPI_Fraud_xgb_model.pkl"),
        "features": [
            "amount", "MerchantCategory", "TransactionType", "Latitude", "Longitude", 
            "AvgTransactionAmount", "TransactionFrequency", "UnusualLocation", 
            "UnusualAmount", "NewDevice", "FailedAttempts", "hour", "day_of_week"
        ],
        "numeric_ranges": {
            "amount": (1.0, 1e9),  # Adjusted max value
            "Latitude": (-90.0, 90.0),
            "Longitude": (-180.0, 180.0),
            "AvgTransactionAmount": (1.0, 1e9),  # Adjusted max value
            "TransactionFrequency": (0, 100),
            "FailedAttempts": (0, 10),
            "hour": (0, 23),
            "day_of_week": (0, 6)
        },
        "categorical_features": ["MerchantCategory", "TransactionType", "UnusualLocation", "UnusualAmount", "NewDevice"]
    },
    "Bank Account Fraud": {
        "rf_model": joblib.load("Bank_Account_Fraud_rf_model.pkl"),
        "xgb_model": joblib.load("Bank_Account_Fraud_xgb_model.pkl"),
        "features": [
            "income", "name_email_similarity", "prev_address_months_count", 
            "current_address_months_count", "customer_age", "days_since_request",
            "amount", "payment_type", "zip_count_4w", "velocity_6h", "velocity_24h",
            "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
            "employment_status", "credit_risk_score", "email_is_free", "housing_status",
            "phone_home_valid", "phone_mobile_valid", "bank_months_count",
            "has_other_cards", "proposed_credit_limit", "foreign_request",
            "device_distinct_emails_8w", "device_fraud_count"
        ],
        "numeric_ranges": {
            "amount": (1.0, 1e6),  # Adjusted max value
            "income": (0, 1e7),  # Adjusted max value
            "customer_age": (18, 100),
            "prev_address_months_count": (0, 360),
            "current_address_months_count": (0, 360),
            "days_since_request": (0, 365),
            "zip_count_4w": (0, 100),
            "velocity_6h": (0, 1000),
            "velocity_24h": (0, 5000),
            "velocity_4w": (0, 10000),
            "bank_branch_count_8w": (0, 100),
            "date_of_birth_distinct_emails_4w": (0, 50),
            "credit_risk_score": (300, 850),
            "bank_months_count": (0, 600),
            "proposed_credit_limit": (500, 1e6),  # Adjusted max value
            "device_distinct_emails_8w": (0, 50),
            "device_fraud_count": (0, 100),
            "name_email_similarity":(0.0,1.0)
        },
        "categorical_features": ["payment_type", "employment_status", "housing_status", "UnusualLocation", "UnusualAmount","phone_home_valid","phone_mobile_valid","foreign_request","has_other_cards","email_is_free"]
    }
}

# Analysis Functions
def get_risk_level(fraud_prob):
    if fraud_prob < 0.3:
        return "Low", "green"
    elif fraud_prob < 0.7:
        return "Medium", "orange"
    else:
        return "High", "red"

def prepare_input_data(user_inputs, feature_list):
    df = pd.DataFrame([user_inputs], columns=feature_list)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    return df

def predict_fraud(input_data, rf_model, xgb_model, feature_list):
    input_df = prepare_input_data(input_data, feature_list)
    rf_pred = rf_model.predict_proba(input_df)[:, 1]
    xgb_pred = xgb_model.predict_proba(input_df)[:, 1]
    final_prob = (rf_pred + xgb_pred) / 2
    return final_prob[0]
# Function to show transaction history
def show_transaction_history():
    st.markdown("## Transaction History")

    # Retrieve user_id from session state
    user_email = st.session_state.get("user")
    if not user_email:
        st.warning("User not authenticated. Please log in.")
        return

    # Fetch user ID from Supabase using email
    user_id = get_user_id(user_email)
    if user_id is None:
        st.error("Could not find user in database.")
        return

    # Fetch transaction history for the logged-in user
    transactions = supabase.table("transactions").select("*").eq("user_id", user_id).execute()
    transactions_df = pd.DataFrame(transactions.data)

    if transactions_df.empty:
        st.info("No transactions found.")
        return

    # Display transaction history in a table
    st.dataframe(transactions_df)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_transaction_history(user_id):
    response = supabase.table("transactions").select("*").eq("user_id", user_id).execute()
    return response.data if response.data else []

def get_user_id(email):
    """Fetch user ID (UUID) from Supabase using email."""
    try:
        response = supabase.table("users").select("user_id").eq("email", email).execute()

        if response.data and len(response.data) > 0:
            return response.data[0]["user_id"]  # Return user_id (UUID)
        else:
            st.error("User not found in the database.")
            return None
    except Exception as e:
        st.error(f"Error fetching user ID: {e}")
        return None
def show_transaction_history():
    st.markdown("## Transaction History")

    user_id = st.session_state.get("user_id")  # Get logged-in user's ID

    if not user_id:
        st.warning("User not authenticated. Please log in.")
        return

    transactions = fetch_transaction_history(user_id)  # Fetch user's transactions

    if not transactions:
        st.info("No transactions found.")
        return

    st.dataframe(transactions)  # Display table
import streamlit as st
from supabase import create_client

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_transaction_history(user_id):
    """Fetches transaction history for the logged-in user from Supabase."""
    if not user_id:
        return []

    response = supabase.table("transactions").select("*").eq("user_id", user_id).execute()
    
    if response.data:
        return response.data
    else:
        return []

def show_analysis_results(fraud_prob, risk_level, risk_color, status, alert_class):
    """Displays transaction analysis results and saves data in Supabase."""

    st.markdown("### Transaction Analysis Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"<div class='{alert_class}'>{status}</div>", unsafe_allow_html=True)
        st.metric("Risk Level", risk_level)
    
    with col2:
        st.metric("Fraud Percentage", f"{fraud_prob:.1%}")
    
    with col3:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 30], 'color': 'lightgreen'},
                    {'range': [30, 70], 'color': 'lightyellow'},
                    {'range': [70, 100], 'color': 'mistyrose'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        gauge.update_layout(height=200)
        st.plotly_chart(gauge)

    # 🔹 Fetch user_id from Supabase using email
    user_email = st.session_state.get("user")

    if not user_email:
        st.error("User email is missing. Please log in again.")
        st.stop()

    user_id = get_user_id(user_email)  # Convert email to user_id

    if user_id is None:
        st.error("Could not find user in database.")
        st.stop()

    # 🔹 Insert transaction using user_id (NOT email)
    try:
        transaction_data = {
            "user_id": user_id,  # ✅ Use user_id instead of email
            "timestamp": datetime.now().isoformat(),
            "transaction_type": menu,  
            "amount": float(user_inputs.get("amount", 0)),
            "fraud_probability": fraud_prob,
            "risk_level": risk_level,
            "model_used": f"{menu}_Model"
        }
        
        response = supabase.table("transactions").insert(transaction_data).execute()
        
        if response.data:
            st.success("Transaction saved successfully!")
        
        else:
            st.success(f"Error saving transaction: {response.__dict__}")

    except Exception as e:
        st.error(f"Error saving transaction: {e}")





def show_visualizations():
    try:
        transactions = supabase.table("Transactions").select("*").execute()
        if transactions.data:
            st.session_state["transactions"] = pd.DataFrame(transactions.data)
            st.dataframe(st.session_state["transactions"])
        else:
            st.warning("No transactions found.")
    except Exception as e:
        st.error(f"Error fetching transactions: {e}")

def show_visualizations():
    if len(st.session_state["transactions"]) > 0:
        st.markdown("### Transaction Analytics")
        
        # Risk Level Distribution
        col1, col2 = st.columns(2)
        with col1:
            risk_dist = st.session_state["transactions"]['Risk_Level'].value_counts()
            fig1 = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                title="Risk Level Distribution",
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            )
            st.plotly_chart(fig1)

        with col2:
            # Fraud Probability Timeline
            fig2 = px.line(
                st.session_state["transactions"],
                x='Timestamp',
                y='Fraud_Probability',
                color='Transaction_Type',
                title="Fraud Probability Timeline"
            )
            st.plotly_chart(fig2)

        # Transaction Amount vs Fraud Probability
        fig3 = px.scatter(
            st.session_state["transactions"],
            x='Amount',
            y='Fraud_Probability',
            color='Risk_Level',
            symbol='Transaction_Type',
            title="Transaction Amount vs Fraud Probability",
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        )
        st.plotly_chart(fig3)

# Main App Layout
if not st.session_state["authenticated"]:
    image_path = "bank.jpg"
    st.sidebar.image(image_path, use_container_width=True)
    auth_choice = st.sidebar.radio("Select:", ["Login", "Signup", "Forgot Password"])
    image_path = "bank.jpg"
    st.sidebar.image(image_path, use_container_width=True)
    if auth_choice == "Signup":
        signup()
    elif auth_choice == "Forgot Password":
        forgot_password()
    else:
        login()
    st.stop()

# Show Logout Button After Login
if st.sidebar.button("Logout"):
    logout()

# Main Navigation
st.sidebar.title("🔍 Fraud Detection System")
image_path = "bank.jpg"
st.sidebar.image(image_path, use_container_width=True)
menu = st.sidebar.radio("Choose Option:", ["Credit Card Fraud", "UPI Fraud", "Bank Account Fraud", "History"])
image_path = "bank.jpg"
st.sidebar.image(image_path, use_container_width=True)
if menu == "History":
    st.subheader("📜 Transaction History")
    if not st.session_state["transactions"].empty:
        st.dataframe(st.session_state["transactions"])
        show_visualizations()
    else:
        st.info("⚠️ No transaction history available.")
    st.stop()

# Transaction Analysis
st.title(f"🚀 {menu} Detection")
selected_model = models[menu]
features = selected_model["features"]

# User Inputs
user_inputs = {}
for feature in features:
    if feature in selected_model["numeric_ranges"]:
        min_val, max_val = selected_model["numeric_ranges"][feature]
        
        # Determine the type of the numeric input (float or int)
        is_float = isinstance(min_val, float) or isinstance(max_val, float)
        
        user_inputs[feature] = st.number_input(
            f"{feature} *", 
            min_value=float(min_val) if is_float else int(min_val), 
            max_value=float(max_val) if is_float else int(max_val), 
            step=0.01 if is_float else 1
        )
    elif feature in selected_model["categorical_features"]:
        if feature in ["employment_status", "housing_status", "foreign_request", "phone_home_valid", "phone_mobile_valid"]:
            user_inputs[feature] = st.selectbox(f"{feature} *", ["Select", "Yes", "No"])
        elif feature == "gender":
            user_inputs[feature] = st.selectbox(f"{feature} *", ["Select", "Male", "Female"])
        elif feature in ["category", "MerchantCategory","merchant"]:
            user_inputs[feature] = st.selectbox(f"{feature} *", ["Select", "Retail", "Service", "E-Commerce", "Utilities", "Other"])
        elif feature in ["UnusualLocation", "UnusualAmount", "NewDevice","phone_home_valid","phone_mobile_valid","foreign_request","has_other_cards","email_is_free"]:
            user_inputs[feature] = st.selectbox(f"{feature} *", ["Select", "Yes", "No"])
        elif feature in ["payment_type","TransactionType"]:
            user_inputs[feature]=st.selectbox(f"{feature} *",["Select","Credit Card", "Debit Card", "Net Banking", "UPI", "Wallet", "Other"])
    else:
        user_inputs[feature] = st.text_input(f"{feature} *", "0")

if st.button("Analyze Transaction"):
    # Check if all required fields are filled
    if any(value in ["", "Select", None] for value in user_inputs.values()):
        st.error("❌ Please fill in all required fields before making a prediction.")
    else:
        fraud_prob = predict_fraud(user_inputs, selected_model["rf_model"], 
                                 selected_model["xgb_model"], features)
        risk_level, risk_color = get_risk_level(fraud_prob)
        
        if risk_level == "High":
            status = "Alert: High Risk Transaction"
            alert_class = "alert-danger"
        elif risk_level == "Medium":
            status = "Caution: Medium Risk Transaction"
            alert_class = "alert-warning"
        else:
            status = "Safe: Low Risk Transaction"
            alert_class = "alert-success"
        
        # Add to transaction history
        new_transaction = {
            "Timestamp": datetime.now(),
            "Transaction_Type": menu,
            "Amount": float(user_inputs.get("amount", 0)),
            "Fraud_Probability": fraud_prob,
            "Risk_Level": risk_level,
            "Model_Used": f"{menu}_Model"
        }
        st.session_state["transactions"] = pd.concat([
            st.session_state["transactions"], 
            pd.DataFrame([new_transaction])
        ], ignore_index=True)

        # Show analysis results
        show_analysis_results(fraud_prob, risk_level, risk_color, status, alert_class)
        show_visualizations()

# CSS styling
st.markdown("""
    <style>
        .alert-danger {
            color: white;
            background-color: red;
            padding: 10px;
            border-radius: 5px;
        }
        .alert-warning {
            color: black;
            background-color: orange;
            padding: 10px;
            border-radius: 5px;
        }
        .alert-success {
            color: white;
            background-color: green;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)