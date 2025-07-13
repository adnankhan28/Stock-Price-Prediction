import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load and preprocess
print("🔄 Loading dataset...")
df = pd.read_csv("APPLE_2006-01-01_to_2018-01-01.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Plot original closing price
print("📊 Plotting original closing price...")
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Close Price', color='blue')
plt.title('Apple Stock Closing Price (2006–2018)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("closing_price_plot.png")
plt.show(block=False)

# Create target column for prediction
future_days = 1
df['Prediction'] = df[['Close']].shift(-future_days)

# Drop rows with NaNs
df.dropna(inplace=True)

# Prepare features and labels
X = df[['Close']].values
y = df['Prediction'].values

print(f"✅ Features shape: {X.shape}")
print(f"✅ Labels shape: {y.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("🧠 Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
confidence = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {confidence * 100:.2f}%")

# Predict next day price
X_future = df[['Close']].values[-1:]
future_price = model.predict(X_future)
print(f"📈 Predicted Next Day Closing Price: ₹{future_price[0]:.2f}")

# Predict on test set
lr_prediction = model.predict(X_test)

# Plot Actual vs Predicted
print("📉 Plotting Actual vs Predicted graph...")
plt.figure(figsize=(12, 6))
plt.plot(y_test[:100], label='Actual Price', color='blue')
plt.plot(lr_prediction[:100], label='Predicted Price', color='orange')  # Limit for visibility
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# Save the model
joblib.dump(model, "stock_price_predictor_model.pkl")
print("💾 Model saved as 'stock_price_predictor_model.pkl'")

import time
time.sleep(5)  # wait 5 seconds before closing all plots
