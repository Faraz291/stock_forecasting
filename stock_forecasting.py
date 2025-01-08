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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import io
from prophet import Prophet


# title
st.title('Stock Forecasting using Machine Learning')
st.subheader('This app is created to forecast stock prices of selected stocks')
# Adding Button
# Markdown styled as a button
st.markdown(
    """
    <style>
    .button {
        display: inline-block;
        padding: 0.5em 1em;
        font-size: 1.2em;
        color: #FFF;
        background-color: white;
        border-radius: 5px;
    }
    </style>
    <a href="http://localhost:8502" target="_self" class="button">Stock Comparison</a>
    """,
    unsafe_allow_html=True
)
# **********************************************************************************************************************************
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
algo = st.sidebar.selectbox('Select the model', ['None', 'ARIMA', 'SARIMA', 'LSTM', 'Prophet'])

def parameter_ui(algo):
    params = dict()
    if algo == 'None':
        st.sidebar.write("Please select the model")
    elif algo == 'ARIMA':
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
        st.sidebar.write("LSTM model is running, it takes time")
    
    elif algo == 'Prophet':
        st.sidebar.write("Prophet algorithm running")
    
    return params
params = parameter_ui(algo)

# ********************************************************************************************************************************************************
# fetch the user selected ticker data
data = yf.download(ticker, start=start_date, end=end_date)

# add date as a column in a dataframe
# data.insert(0, 'Date', data.index, True)
# data.reset_index(drop=True, inplace=True)
# data = data.reset_index()
st.write('Data from', start_date, 'to', end_date)
st.write(data)

# plot the data
st.header('Stock Closing Price (USD)')
# st.subheader('Line Chart')
fig = px.line(data, x=data.index, y=data['Close'].squeeze())
fig.update_layout(title=f'{ticker} Stock Closing Price', xaxis_title='Date', yaxis_title='Price (USD)', width=1100, height=600)

st.plotly_chart(fig)

# Decompose the data
st.header('Decomposition of the data')
# decomposition = seasonal_decompose(data['Close'], model='additive', period=365) # using a fixed seasonal pattern
# st.write(decomposition.plot())

stl = STL(data['Close'], period=365)  # use LOESS (Locally Estimated Scatterplot Smoothing).
result = stl.fit()

# Plot the components
fig = result.plot()
st.pyplot(fig)

# ADF test for check stationarity
st.header('Is data stationary?')
# st.write(adfuller(data['Close']))
st.write('p-value:', adfuller(data['Close'])[1] < 0.05) # if p-value is less than 0.05, then data is stationary

split_index = int(len(data) * 0.8)  # 80-20 chronological split
train_data1 = data[:split_index].Close
test_data1 = data[split_index:].Close
# st.write('Train Data:', train_data1.shape)
# st.write('Test Data:', test_data1.shape)



# ********************************************************************************************************************************************************
# ************************* ARIMA Model *************************

def arima_forecast(forecast_period, params):
    model = sm.tsa.ARIMA(data['Close'], order=(params['p'], params['d'], params['q']))
    model = model.fit()
    # predict the future values
    predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period)  
    predictions = predictions.predicted_mean
    # st.write(predictions)

    # adding index to the predictions
    predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
    predictions = pd.DataFrame(predictions)
    predictions.index.name = 'Date'
    st.write('Actual Data', data['Close'].tail())
    st.write('Predictions', predictions)
    st.write('---')

    # print model summary
    st.header('Model Summary')
    st.write(model.summary())
    st.write('---')

    # plot the forecasted values
    fig = go.Figure()
    # adding actual data
    fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'][-90:].squeeze(), mode='lines', name='Actual Data', line=dict(color='blue')))
    # adding forecasted data
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions['predicted_mean'], mode='lines', name='Forecasted Data', line=dict(color='red')))
    # set the title and axis labels
    fig.update_layout(title=f'{ticker} Stock Forecasting', xaxis_title='Date', yaxis_title='Price (USD)', width=1100, height=600)
    st.plotly_chart(fig)


    # Assigning variables for evaluation
    y_pred = model.predict(start=len(train_data1), end=len(train_data1)+test_data1.shape[0]-1)  
    # print(y_pred.shape)
    y_true = test_data1
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # Display metrics
    st.write("### Model Evaluation")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R-squared (R2):** {r2:.2f}")
    st.write(f"**Mean Absolute Percentage Error (MAE):** {mape:.2f}%")



# ********************************************************************************************************************************************************
# ************************* SARIMA Model *************************

def sarima_forecast(forecast_period, params):   # we call sarima by sarimax... exogenus= false hota hai True krengy tou Sarimax hojaega
    model = model = sm.tsa.statespace.SARIMAX(data['Close'], order=(params['p'], params['d'], params['q']), seasonal_order=params['seasonal_order'])
    model = model.fit()

    # predict the future values
    predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period)  
    predictions = predictions.predicted_mean
    # st.write(predictions)

    # adding index to the predictions
    predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
    predictions = pd.DataFrame(predictions)
    predictions.index.name = 'Date'
    st.write('Actual Data', data['Close'].tail())
    st.write('Predictions', predictions)
    st.write('---')
    # print model summary
    st.header('Model Summary')
    st.write(model.summary())
    st.write('---')
    # plot the forecasted values
    fig = go.Figure()
    # adding actual data
    fig.add_trace(go.Scatter(x=data.index[-90:], y=data['Close'][-90:].squeeze(), mode='lines', name='Actual Data', line=dict(color='blue')))
    # adding forecasted data
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions['predicted_mean'], mode='lines', name='Forecasted Data', line=dict(color='red')))
    # set the title and axis labels
    fig.update_layout(title=f'{ticker} Stock Forecasting', xaxis_title='Date', yaxis_title='Price (USD)', width=1100, height=600)
    st.plotly_chart(fig)


    # Assigning variables for evaluation
    y_pred = model.predict(start=len(train_data1), end=len(train_data1)+test_data1.shape[0]-1) 
    y_true = test_data1
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # Display metrics
    st.write("### Model Evaluation")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R-squared (R2):** {r2:.2f}")
    st.write(f"**Mean Absolute Percentage Error (MAE):** {mape:.2f}%")


# ********************************************************************************************************************************************************
# ************************* LSTM Model *************************

def LSTM_model(forecast_period):
    # .reshape(-1, 1) is used to convert 1D array to 2D array
    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = data['Close']
    # data_scaled = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    data_scaled = scaler.fit_transform(df1.values.reshape(-1, 1))   # values is used to convert series to array
    st.write(f'{ticker} Stock Closing Price')
    st.write(df1) 
    # st.write(data_scaled)

    # # Split the data into training and testing sets
    training_size = int(len(data_scaled) * 0.8)
    test_size = len(data_scaled) - training_size
    train_data, test_data = data_scaled[0:training_size, :], data_scaled[training_size:len(data_scaled), :1]    # :1 is used to convert 1D array to 2D array
    # st.write(len(data_scaled))
    # st.write(len(train_data))
    # st.write(len(test_data))
    # st.write(training_size, test_size)

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
    # st.write(X_train.shape) 
    # st.write(y_train.shape) 
    # st.write(X_test.shape) 
    # st.write(y_test.shape)

    # Reshape data to 3D for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)) # converting 2D to 3D
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)) 


    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))     #X_train.shape[1] means 100 time steps  
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # # Capture the model summary as a string
    # summary_string = io.StringIO()
    # model.summary(print_fn=lambda x: summary_string.write(x + "\n"))
    # summary = summary_string.getvalue()
    # summary_string.close()

    # # Display the model summary
    # st.text("Model Summary:")
    # st.text(summary)
    # st.write(model.summary())

    # # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

    # Lets do the prediction and check performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Assigning variables for evaluation
    y_pred = test_predict
    # st.write('y_test:', y_test)
    # st.write('test_predict:', y_pred)
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    # Display metrics
    st.write("### Model Evaluation")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R-squared (R2):** {r2:.2f}")
    st.write(f"**Mean Absolute Percentage Error (MAE):** {mape:.2f}%")


    # plotting
    look_back = 100
    trainPredictPlot = np.empty_like(data_scaled)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(data_scaled)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(data_scaled)-1, :] = test_predict

    # Create the Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df1.index, y=scaler.inverse_transform(data_scaled).flatten(), mode='lines', name='Original Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df1.index, y=trainPredictPlot.flatten(), mode='lines', name='True Price', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df1.index, y=testPredictPlot.flatten(), mode='lines', name='Predicted Price', line=dict(color='red')))
    fig.update_layout(
        title="Stock Price Prediction",
        xaxis_title="Date",
        yaxis_title="Stock Price in USD",
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig)

    # Forecasting
    x_input=test_data[len(test_data)-look_back:].reshape(1,-1)
    # st.write(x_input.shape)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    # st.write(temp_input)

    lst_output=[]
    n_steps=100
    i=0
    while(i<forecast_period):
        if(len(temp_input)>100):
            x_input=np.array(temp_input[1:])
            # st.write("{} day input {}" .format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input=x_input.reshape((1, n_steps, 1))

            yhat=model.predict(x_input, verbose=0)
            # st.write("{} day output {}" .format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]

            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input=x_input.reshape(1, n_steps, 1)
            yhat=model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1

    # st.write(lst_output)  

    day_pred = pd.date_range(start=end_date, periods=forecast_period, freq='D')
    # st.write(day_pred)

    df3=data_scaled.tolist()
    df3.extend(lst_output)

    st.write(f'{forecast_period} days forecasting of {ticker} Stock')
    # st.write(day_pred, scaler.inverse_transform(lst_output).flatten())
    df_lstm = pd.DataFrame(scaler.inverse_transform(lst_output).flatten(), day_pred)
    df_lstm.reset_index(inplace=True)
    df_lstm = df_lstm.set_index('index')
    df_lstm.index.name = 'Date'
    df_lstm.columns = ['Forecasted Price']
    st.write(df_lstm)

    # Create the Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df1.index[-90:], y=scaler.inverse_transform(data_scaled[len(data_scaled)-100:]).flatten(), mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=day_pred, y=scaler.inverse_transform(lst_output).flatten(), mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(
        title= f'{ticker} Stock Forecasting',
        xaxis_title= "Date",
        yaxis_title="Price (USD)",
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig)


# ********************************************************************************************************************************************************
# ************************* Prophet Model *************************

def prophet(forecast_period):
    df_pro = data['Close']
    # st.header(f'{ticker} Stock Forecasting using Prophet')
    st.write('Original Data')
    st.write(df_pro)

    df1=df_pro.reset_index(inplace=True)
    # st.write(df1)

    df_pro['Date'] = pd.to_datetime(df_pro['Date'])  # Convert to datetime
    df_prophet = df_pro.rename(columns={"Date": "ds", ticker: "y"})  # Prophet expects these column names
    # st.write(df_prophet.columns)

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df_prophet)

    # Create a future dataframe
    future = model.make_future_dataframe(periods=forecast_period)  # Forecast for the next 30 days
    forecast = model.predict(future)

    # Display the forecast data
    st.write("Forecasted Data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period))

    # Plot the forecast
    st.write('Prophet model forecasting')
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.write('Decomposition of the data')
    # Plot the components
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # st.write(forecast)
    # df_pro.set_index('Date')[ticker]


# ********************************************************************************************************************************************************

# predict the future values (forecasting)
forecast_period = st.number_input('Enter the number of days for forecasting', 1, 365, 10)

# applying model
def model_impl(algo, params, forecast_period):
    if algo == 'ARIMA':
        st.header(f'{ticker} Stock Forecasting using ARIMA')
        arima_forecast(forecast_period, params)
    elif algo == 'SARIMA':
        st.header(f'{ticker} Stock Forecasting using SARIMA')
        sarima_forecast(forecast_period, params)
    elif algo == 'LSTM':
        st.header(f'{ticker} Stock Forecasting using LSTM')
        LSTM_model(forecast_period)
    elif algo == 'Prophet':
        st.header(f'{ticker} Stock Forecasting using Prophet')
        prophet(forecast_period)    
    # return model


model_impl(algo, params, forecast_period)


# ********************************************************************************************************************************************************

st.write('---')
# st.write('# THE END')
st.markdown("<h1 style='text-align: center;'>*** THE END ***</h1>", unsafe_allow_html=True)

st.write('---')
st.header("This app developed by: Faraz Ahmed")
st.markdown('<a href="https://github.com/Faraz291/" target="_blank">GitHub</a>', unsafe_allow_html=True)    
st.markdown('<a href="https://www.linkedin.com/in/farazahmed1997/" target="_blank">LinkedIn</a>', unsafe_allow_html=True)
st.markdown('<a href="https://www.instagram.com/faraz__ahmed/" target="_blank">Instagram</a>', unsafe_allow_html=True)
