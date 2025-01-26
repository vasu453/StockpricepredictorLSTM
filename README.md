# Install and import necessary libraries
!pip install yfinance
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import math
# Fetch data for Toyota (TM)
toyota_data = yf.download('TM', start='2015-01-01', end='2024-01-01')

# Display the first few rows
print(toyota_data.head())

# fill missing values 
toyota_data.fillna(method='ffill', inplace=True)

# focus on 'Adj Close' price 
df = toyota_data[['Adj Close']].copy()

# Create Moving Averages as additional features 
df['MA10'] = df['Adj Close'].rolling(window=10).mean()
df['MA50'] = df['Adj Close'].rolling(window=50).mean()

# Drop NaN values created by moving averages 
df.dropna(inplace=True)
# 1. scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Adj Close']])  

# 2. creating features ( 60-day window to predict the next day)
def create_features(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])  # previous 60 days
        y.append(dataset[i, 0])  # current day 
    return np.array(X), np.array(y)

# Prepare the data
X, y = create_features(scaled_data, time_step=60)

# 3. Splitting the data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input data to be 3D for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer
#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Plotting training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
# predictions and RMSE calculation
train_predict = model.predict(X_train)  
test_predict = model.predict(X_test)  

# reshape before inverse transform
train_predict = train_predict.reshape(-1, 1)
test_predict = test_predict.reshape(-1, 1)

# inverse transform the predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# inverse transform the true Values
y_train_inverse = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

# calculate RMSE
train_rmse = math.sqrt(mean_squared_error(y_train_inverse, train_predict))
test_rmse = math.sqrt(mean_squared_error(y_test_inverse, test_predict))

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# visualize the predictions
plt.figure(figsize=(14, 5))
plt.plot(y_test_inverse, label='Actual Prices', color='blue')
plt.plot(test_predict, label='Predicted Prices', color='green')
plt.title('Toyota Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

import pandas as pd

def predict_future_prices(model, df, scaler, n_predictions=5):
    # get the last 60 days of adjusted close prices
    last_60_days = df['Adj Close'][-60:].values
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

    # create a list to hold future predictions
    future_predictions = []

    # start predicting
    current_input = last_60_days_scaled.reshape((1, last_60_days_scaled.shape[0], 1))  

    for _ in range(n_predictions):
        predicted_price = model.predict(current_input)
        future_predictions.append(predicted_price[0][0])  

        # update the current input to include the new predicted price
        current_input = np.append(current_input[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

    # inverse transform the predictions to get them back to the original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # create a list of dates for the predictions
    last_date = df.index[-1]  # get the last date in the dataframe
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_predictions + 1)]

    # create a dataframe with dates and predictions
    predictions_with_dates = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': future_predictions.flatten()
    })
    # At the end of your predict_future_prices function
    predictions_with_dates['Date'] = predictions_with_dates['Date'].dt.date  
    return predictions_with_dates
print("PREPARED THE FUNCTION FUTURE PREDICTIONS!!!")
# function to get the desired amount of days of prediction
ip = int(input("Enter the number of days you want to predict the price after 29-12-2023:"))
n_predictions = ip  
future_prices_with_dates = predict_future_prices(model, df, scaler, n_predictions)
print("Future Price Predictions with Dates:")
print(future_prices_with_dates)


