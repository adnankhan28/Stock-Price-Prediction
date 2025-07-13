import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Predictor App")
st.write("This app predicts the **next day's closing price** using Linear Regression.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("stock_price_predictor_model.pkl")

model = load_model()

# File upload
uploaded_file = st.file_uploader("📤 Upload your stock CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Show raw data
    st.subheader("📄 Raw Data")
    st.dataframe(df.tail(10))

    # Show closing price chart
    st.subheader("📉 Closing Price Over Time")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df['Close'], color='blue', label='Close Price')
    ax.set_title("Stock Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Prepare prediction
    future_days = 1
    df['Prediction'] = df[['Close']].shift(-future_days)
    df.dropna(inplace=True)

    # Features and labels
    X = df[['Close']].values
    y = df['Prediction'].values

    # Predict on recent value
    last_value = np.array(df[['Close']].values[-1]).reshape(1, -1)
    next_day_price = model.predict(last_value)[0]

    st.subheader("💵 Predicted Next Day Closing Price")
    st.success(f"₹ {next_day_price:.2f}")

    # Optional: show Actual vs Predicted on test sample
    st.subheader("📊 Simulated Prediction vs Actual")
    y_pred = model.predict(X)
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(y[:100], label="Actual Price", color='green')
    ax2.plot(y_pred[:100], label="Predicted Price", color='red')
    ax2.set_title("Actual vs Predicted (First 100 samples)")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Price")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
else:
    st.warning("Please upload a stock dataset CSV file.")
