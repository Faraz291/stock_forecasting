import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, time
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# title
st.title('Stock Forecasting using Machine Learning')
st.subheader('This app is created to forecast stock prices of selected stocks')
# image
st.image("D:\python-ka-chilla\project\stock_cover.webp")

# sidebar
st.sidebar.title('Select the parameters from below')

# take user input of start and end date
start_date = st.sidebar.date_input('Start Date', date(2018, 1, 1))
end_date = st.sidebar.date_input('End Date', date.today())

# add ticker list
ticker_list = ['AMZN', 'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'V', 'TSLA', 'META', 'XOM', 'SHEL', 'CVX', 'BYD', 'MA', 'AXP', 'HMC', 'F', 'MSBHF', 'STLA', 
               'VWAGY', 'WBD', 'DIS', 'UVV', 'PARA', 'SONY', 'PYPL']
ticker = st.sidebar.selectbox('Select the stock ticker', ticker_list)

# choose your model
algo = st.sidebar.selectbox('Select the model', ['ARIMA', 'SARIMA', 'LSTM', 'Prophet'])

def parameter_ui(algo):
    params = dict()
    if algo == 'ARIMA':
        p = st.sidebar.slider('Select the value of p', 0, 5, 2)
        d = st.sidebar.slider('Select the value of d', 0, 2, 1)
        q = st.sidebar.slider('Select the value of q', 0, 5, 2)
        params['p'] = p
        params['d'] = d
        params['q'] = q

    elif algo == 'SARIMA':
        p = st.sidebar.slider('Select the value of p', 0, 5, 2)
        d = st.sidebar.slider('Select the value of d', 0, 2, 1)
        q = st.sidebar.slider('Select the value of q', 0, 5, 2)
        P = st.sidebar.slider('Select the seasonal AR order (P)', 0, 5, 1)
        D = st.sidebar.slider('Select the seasonal differencing order (D)', 0, 2, 1)
        Q = st.sidebar.slider('Select the seasonal MA order (Q)', 0, 5, 1)
        m = st.sidebar.number_input('Select the seasonal period (m)', 1, 24, 12)
        params['p'] = p
        params['d'] = d
        params['q'] = q
        params['seasonal_order'] = (P, D, Q, m)

    elif algo == 'LSTM':
        st.sidebar.write("LSTM parameter selection is not implemented yet.")
    
    elif algo == 'Prophet':
        st.sidebar.write("Prophet parameter selection is not implemented yet.")
    
    return params
params = parameter_ui(algo)




# fetch the user selected ticker data
data = yf.download(ticker, start=start_date, end=end_date)

# add date as a column in a dataframe
# data.insert(0, 'Date', data.index, True)
# data.reset_index(drop=True, inplace=True)
# data = data.reset_index()
st.write('Data from', start_date, 'to', end_date)
st.write(data)

# plot the data
st.header('Stock Closing Price')
# st.subheader('Line Chart')
fig = px.line(data, x=data.index, y=data['Close'].squeeze(), title=f'{ticker} Stock Closing Price', width=1100, height=600)
st.plotly_chart(fig)

# Decompose the data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data['Close'], model='additive', period=365)
st.write(decomposition.plot())

stl = STL(data['Close'], period=365)  
result = stl.fit()

# Plot the components
fig = result.plot()
st.pyplot(fig)

# ADF test for check stationarity
st.header('Is data stationary?')
# st.write(adfuller(data['Close']))
st.write('p-value:', adfuller(data['Close'])[1] < 0.05) # if p-value is less than 0.05, then data is stationary

split_index = int(len(data) * 0.8)  # 80-20 chronological split
train_data = data[:split_index].Close
test_data = data[split_index:].Close
# st.write('Train Data:', train_data)
# st.write('Test Data:', test_data)


# # ************************* ARIMA Model *************************

# def arima_forecast(forecast_period, params):
#     model = sm.tsa.ARIMA(data['Close'], order=(params['p'], params['d'], params['q']))
#     model = model.fit()
#     # predict the future values
#     predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period)  
#     # predictions = model.predict(start=len(data), end=len(data) + forecast_period)  
#     predictions = predictions.predicted_mean
#     # st.write(predictions)

#     # adding index to the predictions
#     predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
#     predictions = pd.DataFrame(predictions)
#     predictions.index.name = 'Date'
#     st.write('Predictions', predictions)
#     st.write('Actual Data', data['Close'].tail())
#     st.write('---')
#     # print model summary
#     st.header('Model Summary')
#     st.write(model.summary())
#     st.write('---')
#     # plot the forecasted values
#     fig = go.Figure()
#     # adding actual data
#     fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'][-90:].squeeze(), mode='lines', name='Actual Data', line=dict(color='blue')))
#     # adding forecasted data
#     fig.add_trace(go.Scatter(x=predictions.index, y=predictions['predicted_mean'], mode='lines', name='Forecasted Data', line=dict(color='red')))
#     # set the title and axis labels
#     fig.update_layout(title=f'{ticker} Stock Forecasting', xaxis_title='Date', yaxis_title='Price', width=1100, height=600)
#     st.plotly_chart(fig)


#     # Assigning variables for evaluation
#     forecast = model.get_forecast(steps=len(test_data))
#     y_pred = forecast.predicted_mean
#     y_true = test_data
#     # Calculate evaluation metrics
#     mae = mean_absolute_error(y_true, y_pred)
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_true, y_pred)

#     # Display metrics
#     st.write("### Evaluation Metrics")
#     st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
#     st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
#     st.write(f"**Mean Absolute Percentage Error (MAPE):** {mse:.2f}")
#     st.write(f"**R-squared (R2):** {r2:.2f}")


# # ************************* SARIMA Model *************************

# def sarima_forecast(forecast_period, params):   # we call sarima by sarimax... exogenus= false hota hai True krengy tou Sarimax hojaega
#     model = model = sm.tsa.statespace.SARIMAX(data['Close'], order=(params['p'], params['d'], params['q']), seasonal_order=params['seasonal_order'])
#     model = model.fit()

#     # predict the future values
#     predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period)  
#     predictions = predictions.predicted_mean
#     # st.write(predictions)

#     # adding index to the predictions
#     predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
#     predictions = pd.DataFrame(predictions)
#     predictions.index.name = 'Date'
#     st.write('Predictions', predictions)
#     st.write('Actual Data', data['Close'].tail())
#     st.write('---')
#     # print model summary
#     st.header('Model Summary')
#     st.write(model.summary())
#     st.write('---')
#     # plot the forecasted values
#     fig = go.Figure()
#     # adding actual data
#     fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'][-90:].squeeze(), mode='lines', name='Actual Data', line=dict(color='blue')))
#     # adding forecasted data
#     fig.add_trace(go.Scatter(x=predictions.index, y=predictions['predicted_mean'], mode='lines', name='Forecasted Data', line=dict(color='red')))
#     # set the title and axis labels
#     fig.update_layout(title=f'{ticker} Stock Forecasting', xaxis_title='Date', yaxis_title='Price', width=1100, height=600)
#     st.plotly_chart(fig)


#     # Assigning variables for evaluation
#     forecast = model.get_forecast(steps=len(test_data))
#     y_pred = forecast.predicted_mean
#     y_true = test_data
#     # Calculate evaluation metrics
#     mae = mean_absolute_error(y_true, y_pred)
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_true, y_pred)

#     # Display metrics
#     st.write("### Evaluation Metrics")
#     st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
#     st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
#     st.write(f"**Mean Absolute Percentage Error (MAPE):** {mse:.2f}")
#     st.write(f"**R-squared (R2):** {r2:.2f}")



# predict the future values (forecasting)
forecast_period = st.number_input('Enter the number of days for forecasting', 1, 365, 10)

# # applying model
# def model_impl(algo, params, forecast_period):
#     if algo == 'ARIMA':
#         arima_forecast(forecast_period, params)
#     elif algo == 'SARIMA':
#         sarima_forecast(forecast_period, params)
#     elif algo == 'LSTM':
#         st.write("LSTM model is not implemented yet.")
#     # return model


# model_impl(algo, params, forecast_period)


# st.plotly_chart(px.line(x=[1, 2, 3], y=[10, 20, 30]))   # just for testing


# ************************* LSTM Model *************************

# # Preprocess data
# scaler = MinMaxScaler(feature_range=(0, 1))
# data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# # Prepare training and test datasets
# def prepare_data(data, time_steps):
#     X, y = [], []
#     for i in range(time_steps, len(data)):
#         X.append(data[i-time_steps:i, 0])
#         y.append(data[i, 0])
#     return np.array(X), np.array(y)

# TIME_STEPS = 6
# X, y = prepare_data(data_scaled, TIME_STEPS)

# # Split the data into training and testing sets
# split = int(0.7 * len(X))   # chronological split
# X_train, y_train = X[:split], y[:split]
# X_test, y_test = X[split:], y[split:]

# # Reshape data to 3D for LSTM input
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
# # st.write(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# # st.write(X_train[1])
# # Build the RNN model
# model = Sequential([
#     LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
#     Dropout(0.2),
#     LSTM(50, return_sequences=False),
#     Dropout(0.2),
#     Dense(25, activation='relu'),
#     Dense(1)
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# # Make predictions
# predictions = model.predict(X_test)
# # predictions = model.predict(forecast_period)
# # predictions = model.predict(start=len(data), end=len(data) + forecast_period)
# st.write(predictions)

# # Reverse scaling to get actual prices
# predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
# y_test = scaler.inverse_transform(y_test.reshape(-1, 1))


# # Plot results using Plotly
# fig = go.Figure()
# fig.add_trace(go.Scatter(y=y_test.flatten(), mode='lines', name='True Price'))
# fig.add_trace(go.Scatter(y=predictions.flatten(), mode='lines', name='Predicted Price'))
# fig.update_layout(
#     title='Stock Price Prediction using LSTM',
#     xaxis_title='Time',
#     yaxis_title='Stock Price',
#     legend=dict(x=0, y=1)
# )
# st.plotly_chart(fig)
# # Forecast future values
# # forecast_period = st.number_input('Enter the number of days for forecasting', 1, 365, 10)
# last_sequence = data_scaled[-TIME_STEPS:]

# future_predictions = []
# current_input = last_sequence
# for _ in range(forecast_period):
#     current_input = current_input.reshape((1, TIME_STEPS, 1))
#     next_pred = model.predict(current_input)
#     future_predictions.append(next_pred[0, 0])
#     current_input = np.append(current_input[0, 1:], next_pred, axis=0)

# # Reverse scaling for forecasted values
# future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
# # st.write(future_predictions)
# # adding index to the predictions
# future_predictions = pd.DataFrame(future_predictions)
# future_predictions.index.name = 'Date'
# future_predictions.index = pd.date_range(start=end_date, periods=len(future_predictions), freq='D')
# future_predictions.rename(columns={0: 'Future Predictions'}, inplace=True)
# st.write(f"Forecasted Prices for the next {forecast_period} days:", future_predictions)
# st.write('Actual Data', data['Close'].tail())
# st.write('---')
# # Display forecasted results
# # st.write(f"Forecasted Prices for the next {forecast_period} days:", future_predictions)
#     # plot the forecasted values
# fig = go.Figure()
# # adding actual data
# fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'][-90:].squeeze(), mode='lines', name='Actual Data', line=dict(color='blue')))
# # adding forecasted data
# fig.add_trace(go.Scatter(x=future_predictions.index, y=future_predictions['Future Predictions'], mode='lines', name='Forecasted Data', line=dict(color='red')))
# # set the title and axis labels
# fig.update_layout(title=f'{ticker} Stock Forecasting', xaxis_title='Date', yaxis_title='Price', width=1100, height=600)
# st.plotly_chart(fig)


# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# # Split the data into training and testing sets
# st.write(len(data_scaled))
# st.write(len(train_data))
# st.write(len(test_data))

# Prepare training and test datasets
def prepare_data(data, time_step=1):
    dataX, dataY = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(data[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = prepare_data(train_data, time_step)
X_test, y_test = prepare_data(test_data, time_step)
st.write(X_train) 
# st.write(Y_train.shape) 
# st.write(X_test.shape) 
# st.write(Y_test.shape)

# # Reshape data to 3D for LSTM input
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(50,return_sequences=True, input_shape=(X_train.shape[1],1)))     #X_train.shape[1] which is 100
# model.add(LSTM(50, return_sequences=True))
# model.add(LSTM(50))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# st.write(model.summary())

