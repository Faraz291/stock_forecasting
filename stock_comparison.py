import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

start_date = "2018-07-01"
end_date = "2024-06-30"
tickers = ["AMZN", 'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'V', 'TSLA', 'META', 'XOM', 'SHEL', 'CVX', 'BYD', 'MA', 
           'AXP', 'HMC', 'F', 'MSBHF', 'STLA', 'VWAGY', 'WBD', 'DIS', 'UVV', 'PARA', 'SONY', 'PYPL'
#          Honda HMC, mistsubishi MSBHF, Ford F, Stellantis STLA, Volkswagen VWAGY, Warner Bros WBD, Walt Disney DIS
#          Universal Pictures UVV, Paramount Pictures PARA, Sony Group SONY, 
          ]

AMZN = yf.download(tickers[0], start=start_date, end=end_date)
AAPL = yf.download(tickers[1], start=start_date, end=end_date)
MSFT = yf.download(tickers[2], start=start_date, end=end_date)
GOOGL = yf.download(tickers[3], start=start_date, end=end_date)
NVDA = yf.download(tickers[4], start=start_date, end=end_date)
V = yf.download(tickers[5], start=start_date, end=end_date)
PYPL = yf.download(tickers[24], start=start_date, end=end_date)
TSLA = yf.download(tickers[6], start=start_date, end=end_date)
META = yf.download(tickers[7], start=start_date, end=end_date)
SHEL = yf.download(tickers[9], start=start_date, end=end_date)
XOM = yf.download(tickers[8], start=start_date, end=end_date)
CVX = yf.download(tickers[10], start=start_date, end=end_date)
BYD = yf.download(tickers[12], start=start_date, end=end_date)
MA = yf.download(tickers[13], start=start_date, end=end_date)
AXP = yf.download(tickers[14], start=start_date, end=end_date)
HMC = yf.download(tickers[15], start=start_date, end=end_date)
F = yf.download(tickers[16], start=start_date, end=end_date)
MSBHF = yf.download(tickers[17], start=start_date, end=end_date)
STLA = yf.download(tickers[18], start=start_date, end=end_date)
VWAGY = yf.download(tickers[19], start=start_date, end=end_date)
WBD = yf.download(tickers[20], start=start_date, end=end_date)
DIS = yf.download(tickers[21], start=start_date, end=end_date)
UVV = yf.download(tickers[22], start=start_date, end=end_date)
PARA = yf.download(tickers[23], start=start_date, end=end_date)
SONY = yf.download(tickers[11], start=start_date, end=end_date)


# filter only 'Close' column and then check null values
dataframes = [AMZN, AAPL, MSFT, GOOGL, NVDA, V, TSLA, META, XOM, SHEL, BYD, MA, AXP, HMC, F, MSBHF, STLA, VWAGY, 
              WBD, DIS, UVV, PARA, SONY, CVX, PYPL]

columns = [df['Close'] for df in dataframes]
# print(columns)
result = pd.concat(columns, axis=1)

# Renaming a column
result.columns = tickers
# print(result)
# print(result.isna().sum())

# making a dataframe of only close prices
stock_close = pd.DataFrame(result)

# making a dataframe of volume
stock_tech = stock_close[['AMZN', 'MSFT', 'META', 'GOOGL', 'NVDA', 'AAPL']]
stock_gasoline = stock_close[['SHEL', 'XOM', 'CVX']]
stock_motor = stock_close[['F', 'MSBHF', 'STLA', 'VWAGY', 'HMC', 'TSLA', 'BYD']]
stock_payment = stock_close[['V', 'AXP', 'MA']]
stock_production = stock_close[['WBD', 'DIS', 'UVV', 'PARA', 'SONY']]

# ***************************************************************************************************************************
# Sidebar
st.sidebar.title('**Stock Name**')
st.sidebar.write('**AMZN:** Amazon')
st.sidebar.write('**MSFT:** Microsoft')
st.sidebar.write('**META:** Meta Technology')
st.sidebar.write('**GOOGL:** Google')
st.sidebar.write('**NVDA:** Nvidia')
st.sidebar.write('**AAPL:** Apple')
st.sidebar.write('**SHEL:** Shell')
st.sidebar.write('**XOM:** Exxon Mobil')
st.sidebar.write('**CVX:** Chevron Corporation')
st.sidebar.write('**F:** Ford')
st.sidebar.write('**MSBHF:** Mitsubishi')
st.sidebar.write('**STLA:** Stellantis')
st.sidebar.write('**VWAGY:** Volkswagen')
st.sidebar.write('**HMC:** Honda')
st.sidebar.write('**TSLA:** Tesla')
st.sidebar.write('**BYD:** Build Your Dreams')
st.sidebar.write('**V:** Visa')
st.sidebar.write('**AMX:** American Express')
st.sidebar.write('**MA:** Mastercard')
st.sidebar.write('**WBD:** Warner Bros')
st.sidebar.write('**DIS:** Walt Disney')
st.sidebar.write('**UVV:** Universal Pictures')
st.sidebar.write('**PARA:** Paramount Pictures')
st.sidebar.write('**SONY:** Sony Group')

# Main Body
# title
st.title('Comparison of Stock Prices')

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
    <a href="http://localhost:8501" target="_self" class="button">Stock Forecasting</a>
    """,
    unsafe_allow_html=True
)


# Figure of Tech Stocks
fig1 = px.line(stock_tech, x=stock_tech.index, y=stock_tech.columns, 
               title='Tech Stock  Price', width=1100, height=550)
st.plotly_chart(fig1)

# # Figure of Gasoline stock
fig2 = px.line(stock_gasoline, x=stock_gasoline.index, y=stock_gasoline.columns, 
               title='Gasoline Stock Price', width=1100, height=550)
st.plotly_chart(fig2)
    
# # Figure of Motor companies stock    
fig3 = px.line(stock_motor, x=stock_motor.index, y=stock_motor.columns, 
               title='Motor Stock Price', width=1100, height=550)
st.plotly_chart(fig3)
    
# # Figure of Payment method's stock    
fig4 = px.line(stock_payment, x=stock_payment.index, y=stock_payment.columns, 
               title='Payment Methods Stock Price', width=1100, height=550)
st.plotly_chart(fig4)
    
# # Figure of Film production's stock    
fig5 = px.line(stock_production, x=stock_production.index, y=stock_production.columns, 
               title='Film Production Stock Price', width=1100, height=550)
st.plotly_chart(fig5)