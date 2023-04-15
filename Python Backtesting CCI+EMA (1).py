#!/usr/bin/env python
# coding: utf-8

# # Downloading Important libraries

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().system('pip install ta')
from ta.momentum import RSIIndicator
import pickle
from datetime import datetime, timedelta
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from nsetools import Nse


# # Downloading the data
# The code defines a function download_stock_data() that takes three parameters - symbol, start_date, and end_date. It uses the Nse module to fetch stock data for the given symbols (companies) from the National Stock Exchange (NSE) of India, within the specified date range. The downloaded data is stored in a dictionary with symbols as keys and stock data as values. The function returns the dictionary. The NIFTY50 companies' symbols are read from a CSV file, and the function is called with the symbols, start date as '2000-01-01', and end date as '2023-04-14' to download the stock data for these companies.

# In[4]:


def download_stock_data(symbol, start_date, end_date):
    nse = Nse()
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    stock_data_dict = {} # Created a dictonary to store the values
    for ticker in symbol:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        print(f"Downloaded data for {ticker}:") # {ticker} print the name of companies 
        print(stock_data)
        stock_data_dict[ticker] = stock_data
    return stock_data_dict

nifty_50 = pd.read_csv("C:/Users/vmik5/OneDrive - Shiv Nadar University/Desktop/data/Webscraping/CATmater/data.csv") # NIFTY50 compaines are saved in CSV file
symbol = nifty_50['Symbol'].tolist()
start_date = '2000-01-01'
end_date = '2023-04-14'
stock_data_dict = download_stock_data(symbol, start_date, end_date)

This code iterates through a dictionary of stock data for different tickers. For each ticker, it adds a new column to the DataFrame with a sequence of numbers representing days, and then prints the relevant stock data (columns: days, Open, High, Low, Close, Volume) for that ticker.
# In[5]:


for ticker, stock_df in stock_data_dict.items(): #ticker is used as the variable name for the keys and stock_df is used as the variable name for the values
    data1 = pd.Series(range(1, len(stock_df) + 1), name='days')
    stock_df = stock_df.join(data1)
    print(f"Stock Data for {ticker}:")
    print(stock_df[["days", "Open", "High", "Low", "Close", "Volume"]].head()) # Taking out only relevant data 


# # Plotting
# This code defines a function called plot_stock_data that takes a dictionary of stock data. It plots the 'Close' and 'Volume' columns of the stock data for the last 365 rows for each ticker in the dictionary, and displays the plots with titles and legends using matplotlib.

# In[6]:


def plot_stock_data(stock_data_dict):
    for ticker, stock_df in stock_data_dict.items():
        plt.figure(figsize=(12, 4))
        stock_df[["Close", "Volume"]][-365:].plot(kind='line', subplots=True, label=['Close', 'Volume'])
        plt.title(f"Stock Data for {ticker}")
        plt.legend(loc=2)
        plt.show()
plot_stock_data(stock_data_dict)


# # Finding EMA's
# This code defines a function called EMA_stock_data that calculates Exponential Moving Averages (EMA) for given periods (ema_periods) on the 'Close' column of stock data for each ticker in a dictionary (stock_data_dict). The calculated EMAs are added as new columns to the stock data DataFrame, and the updated dictionary is returned.

# In[7]:


def EMA_stock_data(stock_data_dict, ema_periods):
    for ticker, stock_df in stock_data_dict.items():
        for i in ema_periods:
            ema_column_name = ("EMA_" + str(i))
            stock_df[ema_column_name] = stock_df['Close'].ewm(span=i, adjust=False).mean()
        
#         stock_df['Company'] = ticker  

    return stock_data_dict

ema_periods = [13, 21, 34] 
EMA_stock_data(stock_data_dict, ema_periods)


# # Plotting EMA's and Closing Price

# In[8]:



def plot_ema(stock_data_dict, tickers, ema_periods):
    for i, ticker in enumerate(tickers):
        data = stock_data_dict[ticker]
        plt.figure(figsize=(12, 4))
        plt.plot(data['Close'][-365:], label=ticker + " CLOSE PRICE")
        for period in ema_periods:
            ema_column_name = "EMA_" + str(period)
            plt.plot(data[ema_column_name][-365:], label=ticker + " EMA_" + str(period))
        plt.title(f"Exponential Moving Averages for {ticker}")
        plt.legend(loc=3)
        plt.show()
tickers = ['HCLTECH.NS', 'HINDALCO.NS', 'ULTRACEMCO.NS','UPL.NS','JSWSTEEL.NS','HDFCLIFE.NS','NESTLEIND.NS','TATACONSUM.NS','TATASTEEL.NS','GRASIM.NS','KOTAKBANK.NS','HDFC.NS','HDFCBANK.NS','ICICIBANK.NS','DRREDDY.NS','ADANIENT.NS','BHARTIARTL.NS','INFY.NS','BRITANNIA.NS','SBIN.NS','AXISBANK.NS','SBILIFE.NS','WIPRO.NS','LT.NS','TECHM.NS','BAJAJFINSV.NS','TATAMOTORS.NS','APOLLOHOSP.NS','ONGC.NS','BAJFINANCE.NS','BAJAJ-AUTO.NS','TITAN.NS','M&M.NS','ADANIPORTS.NS','INDUSINDBK.NS','BPCL.NS','COALINDIA.NS','RELIANCE.NS','HINDUNILVR.NS','DIVISLAB.NS','TCS.NS','HEROMOTOCO.NS','SUNPHARMA.NS','ASIANPAINT.NS','CIPLA.NS','POWERGRID.NS','ITC.NS','MARUTI.NS','NTPC.NS','EICHERMOT.NS']
plot_ema(stock_data_dict, tickers, ema_periods)


# # Calculating CCI Indicator
# calculate_cci(stock_data_dict, window=20): This function calculates the Commodity Channel Index (CCI) indicator for each stock in the stock_data_dict dictionary using the 'High', 'Low', and 'Close' columns. It uses a given window (default value is 20) for Simple Moving Averages (SMA) of the typical price and mean deviation.
# 
# plot_cci(stock_data_dict, tickers): This function plots the CCI indicator values for a list of tickers from the stock_data_dict dictionary. It uses Matplotlib library to create line plots of CCI values for the last 365 days, along with upper and lower limits represented by horizontal lines.

# In[9]:



def calculate_cci(stock_data_dict, window=20):
    cci_dict = {} # this is an empty dictionary which will store the values of cci indicator
    for stock_symbol, stock_df in stock_data_dict.items(): # stock_df contains the values
        typical_price = (stock_df['High'] + stock_df['Low'] + stock_df['Close']) / 3
        sma_typical_price = typical_price.rolling(window=window).mean() # window is already assigned which is 20 
        mean_deviation = abs(typical_price - sma_typical_price).rolling(window=window).mean()
        cci = (typical_price - sma_typical_price) / (0.015 * mean_deviation)
        stock_df['CCI'] = cci  
        cci_dict[stock_symbol] = stock_df
    return cci_dict
calculate_cci(stock_data_dict)

def plot_cci(stock_data_dict, tickers):
    for i, ticker in enumerate(tickers):
        data = stock_data_dict[ticker]
        plt.figure(figsize=(12,4))
        plt.plot(data['CCI'][-365:], label=ticker) #  plotting the values of the 'CCI' column for the last 365 days to avoid cluttering
        plt.axhline(y=100, linestyle=':', color='black',label="Upper Limit")
        plt.axhline(y=-100, linestyle=':', color='blue',label="Lower Limit")
        plt.title(f"CCI Indicator for {ticker}")
        plt.legend(loc=2)
        plt.show()

tickers=['HCLTECH.NS', 'HINDALCO.NS', 'ULTRACEMCO.NS','UPL.NS','JSWSTEEL.NS','HDFCLIFE.NS','NESTLEIND.NS','TATACONSUM.NS','TATASTEEL.NS','GRASIM.NS','KOTAKBANK.NS','HDFC.NS','HDFCBANK.NS','ICICIBANK.NS','DRREDDY.NS','ADANIENT.NS','BHARTIARTL.NS','INFY.NS','BRITANNIA.NS','SBIN.NS','AXISBANK.NS','SBILIFE.NS','WIPRO.NS','LT.NS','TECHM.NS','BAJAJFINSV.NS','TATAMOTORS.NS','APOLLOHOSP.NS','ONGC.NS','BAJFINANCE.NS','BAJAJ-AUTO.NS','TITAN.NS','M&M.NS','ADANIPORTS.NS','INDUSINDBK.NS','BPCL.NS','COALINDIA.NS','RELIANCE.NS','HINDUNILVR.NS','DIVISLAB.NS','TCS.NS','HEROMOTOCO.NS','SUNPHARMA.NS','ASIANPAINT.NS','CIPLA.NS','POWERGRID.NS','ITC.NS','MARUTI.NS','NTPC.NS','EICHERMOT.NS']
plot_cci(stock_data_dict, tickers)


# # Removing NAN
# This code iterates through each key-value pair in the stock_data_dict dictionary. For each key (ticker symbol), it drops any rows with missing values (NaN) from the corresponding DataFrame value. The modified dictionary is then printed inside the loop. Finally, the updated stock_data_dict is printed again outside the loop. In short, this code removes rows with missing values from each DataFrame in the stock_data_dict dictionary.

# In[10]:


for key, value in stock_data_dict.items():
    stock_data_dict[key] = value.dropna() 
    print(stock_data_dict)
print(stock_data_dict)


# # Calculating CCI Indicator
# The function calculate_adx takes a dictionary of stock data as input, along with an optional window parameter for the smoothing period (default is 14). It calculates the Average Directional Index (ADX) for each stock in the dictionary using the True Range, Directional Movement, and Smoothed True Range. The calculated ADX values are added as columns to the input stock data and the updated data is returned as a dictionary with stock symbols as keys.

# In[12]:


def calculate_adx(stock_data_dict, window=14):
    adx_dict = {}
    for stock_symbol, stock_df in stock_data_dict.items():
        high = stock_df['High']
        low = stock_df['Low']
        close = stock_df['Close']

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Directional Movement
        tr = true_range  # True Range
        dm_plus = (high - high.shift(1)).clip(lower=0) / tr
        dm_minus = (low.shift(1) - low).clip(lower=0) / tr

        # Calculate Smoothed True Range and Directional Movement
        atr = true_range.rolling(window=window).mean()
        dm_plus_smooth = dm_plus.rolling(window=window).mean()
        dm_minus_smooth = dm_minus.rolling(window=window).mean()

        # Calculate Positive Directional Index (DI+) and Negative Directional Index (DI-)
        di_plus = 100 * dm_plus_smooth
        di_minus = 100 * dm_minus_smooth

        # Calculate Average Directional Index (ADX)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=window).mean()

        stock_df['DI+'] = di_plus
        stock_df['DI-'] = di_minus
        stock_df['ADX'] = adx

        adx_dict[stock_symbol] = stock_df

    return adx_dict
calculate_adx(stock_data_dict,window=14)


# # PLotting ADX
# The function plot_adx takes in a dictionary stock_data_dict containing stock data for different tickers and a list of tickers tickers. It then plots the Average Directional Index (ADX) for each ticker using the matplotlib library. The plot is limited to the last 365 days of data. A threshold of 25 is also plotted as a horizontal line. The function titles the plot with the ticker name and adds a legend. Finally, the function is called with a list of tickers which contains all the .

# In[13]:


def plot_adx(stock_data_dict, tickers):
    for i, ticker in enumerate(tickers):
        data = stock_data_dict[ticker]
        plt.figure(figsize=(12,4))
        plt.plot(data['ADX'][-365:], label=ticker+" "+ "ADX") # plotting only for last 365 days
        plt.axhline(y=25, linestyle=':', color='Black', label='ADX threshold(25)')
        plt.title(f"Average Directional Index (ADX) for {ticker}")
        plt.legend()
tickers=['HCLTECH.NS', 'HINDALCO.NS', 'ULTRACEMCO.NS','UPL.NS','JSWSTEEL.NS','HDFCLIFE.NS','NESTLEIND.NS','TATACONSUM.NS','TATASTEEL.NS','GRASIM.NS','KOTAKBANK.NS','HDFC.NS','HDFCBANK.NS','ICICIBANK.NS','DRREDDY.NS','ADANIENT.NS','BHARTIARTL.NS','INFY.NS','BRITANNIA.NS','SBIN.NS','AXISBANK.NS','SBILIFE.NS','WIPRO.NS','LT.NS','TECHM.NS','BAJAJFINSV.NS','TATAMOTORS.NS','APOLLOHOSP.NS','ONGC.NS','BAJFINANCE.NS','BAJAJ-AUTO.NS','TITAN.NS','M&M.NS','ADANIPORTS.NS','INDUSINDBK.NS','BPCL.NS','COALINDIA.NS','RELIANCE.NS','HINDUNILVR.NS','DIVISLAB.NS','TCS.NS','HEROMOTOCO.NS','SUNPHARMA.NS','ASIANPAINT.NS','CIPLA.NS','POWERGRID.NS','ITC.NS','MARUTI.NS','NTPC.NS','EICHERMOT.NS']
plot_adx(stock_data_dict,tickers)


# #  Conditions
# The apply_conditions_to_stocks function takes a dictionary of stock data as input, where the keys are stock tickers and the values are pandas DataFrames containing stock data. The function applies a set of conditions to each stock's DataFrame to generate signals for buying or selling opportunities.
# 
# 
# The conditions applied in the function are as follows:
# 
# Buy Signal:
# 
# EMA (Exponential Moving Average) 13-day value is greater than EMA 21-day value.
# EMA 21-day value is greater than EMA 34-day value.
# CCI (Commodity Channel Index) value is less than -100, indicating an extremely oversold condition.
# ADX (Average Directional Index) value is greater than 25, indicating a potential reversal to the upside.
# 
# 
# 
# Sell Signal:
# 
# EMA 13-day value is less than EMA 21-day value.
# EMA 21-day value is greater than EMA 34-day value.
# CCI value is greater than 100, indicating an extremely overbought condition.
# ADX value is greater than 25, indicating a potential reversal to the downside.
# 
# 
# If all of the conditions in the respective buy or sell signal are met for a particular stock, the function assigns a "Buy" or "Sell" signal, respectively, to the corresponding row in the DataFrame. If none of the conditions are met, the "signal" column in the DataFrame remains empty.
# 
# 
# 
# The function uses numpy's np.select() method to apply the conditions efficiently and update the "signal" column in each stock's DataFrame. The modified stock data dictionary with the added "signal" column is returned as the output.
# 
# 
# 
# 
# 

# In[14]:


def apply_conditions_to_stocks(stock_data_dict):
    for ticker, stock_df in stock_data_dict.items():
        close_price = stock_df["Close"]
        ema_13_values = stock_df['EMA_13']
        ema_21_values = stock_df['EMA_21']
        ema_34_values = stock_df['EMA_34']
        cci_values = stock_df['CCI']
        adx_values = stock_df['ADX']
        choices = ["Buy", "Sell"]

        conditions = [            
            (ema_13_values > ema_21_values) &
            (ema_21_values > ema_34_values) &
            (cci_values < -100)&  #A CCI value of -100 typically indicates an extremely oversold condition, 
            (adx_values > 25),     #which means that the price has fallen significantly and may be due for a 
                                 #potential reversal to the upside. This may be interpreted as a buying opportunity.
                      

            (ema_13_values < ema_21_values) &
            (ema_21_values > ema_34_values) &
            (cci_values > 100) &  #CCI value of 100 generally indicates an extremely overbought condition,
             (adx_values > 25)    #which means that the price has risen significantly and may be due for a 
                                  #potential reversal to the downside. This may be interpreted as a selling opportunity.
            
        ]
        
        stock_data_dict[ticker].loc[:, 'signal'] = np.select(conditions, choices, default=None)
    
    return stock_data_dict

result_data_dict = apply_conditions_to_stocks(stock_data_dict)



# # Entry and Exit
# The given code iterates through a list of stock names and applies conditions to filter stock data. It then uses a loop to iterate through the filtered data for each stock. For each stock, it keeps track of buy and sell dates and prices based on the conditions met in the data. The 'position' variable is used to determine if the code is currently in a trade or not. The 'buydates', 'buyprices', 'selldates', and 'sellprices' lists are populated with the corresponding dates and prices based on the conditions met. Finally, the code prints the stock name, buy dates, buy prices, sell dates, and sell prices for each stock.

# In[16]:


stock_names = ['HCLTECH.NS', 'HINDALCO.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'JSWSTEEL.NS', 'HDFCLIFE.NS', 'NESTLEIND.NS',
              'TATACONSUM.NS', 'TATASTEEL.NS', 'GRASIM.NS', 'KOTAKBANK.NS', 'HDFC.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
              'DRREDDY.NS', 'ADANIENT.NS', 'BHARTIARTL.NS', 'INFY.NS', 'BRITANNIA.NS', 'SBIN.NS', 'AXISBANK.NS',
              'SBILIFE.NS', 'WIPRO.NS', 'LT.NS', 'TECHM.NS', 'BAJAJFINSV.NS', 'TATAMOTORS.NS', 'APOLLOHOSP.NS',
              'ONGC.NS', 'BAJFINANCE.NS', 'BAJAJ-AUTO.NS', 'TITAN.NS', 'M&M.NS', 'ADANIPORTS.NS', 'INDUSINDBK.NS',
              'BPCL.NS', 'COALINDIA.NS', 'RELIANCE.NS', 'HINDUNILVR.NS', 'DIVISLAB.NS', 'TCS.NS', 'HEROMOTOCO.NS',
              'SUNPHARMA.NS', 'ASIANPAINT.NS', 'CIPLA.NS', 'POWERGRID.NS', 'ITC.NS', 'MARUTI.NS', 'NTPC.NS', 'EICHERMOT.NS']

result_data_dict = apply_conditions_to_stocks(stock_data_dict)

for stock_name in stock_names:
    data = result_data_dict[stock_name]
    data["signal"] = data["signal"].shift()
    position = False               # here we are making position false as you are not in trade, once we will be in trade position will be True
    buyprices, sellprices = [], []  # empty list for buyprices and sellprices
    buydates, selldates = [], []   # empty list for buydates and selldates
    data['shifted_Close'] = data['Close'].shift()

    for index, row in data.iterrows():               #  we are doing to iter to all the rows 
        if not position and row['signal'] == 'Buy':  # we are doing "not postion"  because we will enter into trade only if we are not into trades
            buydates.append(index)                   # it is appending all the buydates to the list
            buyprices.append(row['Open'])            # it is appending all the buyprices-- which are open prices
            position = True                          # over here position will be true only if we are in trade 
        if position:
            if row['signal'] == 'Sell' or row['shifted_Close'] < 0.90 * buyprices[-1]: # < 0.90 * buyprices[-1]: This is checking if the value of 'shifted_Close' is less than 90% of the value of the last element in the 'buyprices' list, which is indexed by [-1]. It's a comparison operation that evaluates to True if the condition is met, and False otherwise.
                selldates.append(index)              # it is appending all the selldates to the list
                sellprices.append(row['Close'])       # it is appending all the sellprices-- which are Close prices
                position = False                      # over here position will be false  only if we are out of  trade

    print("Stock Name: ", stock_name)
    print("Buy Dates: ", buydates)
    print("Buy Prices: ", buyprices)
    print("Sell Dates: ", selldates)
    print("Sell Prices: ", sellprices)
    print("------------")


# # Plotting Entry and Exit 
# The given code iterates through a dictionary of stock data, extracts the buy and sell dates based on the 'signal' column in the stock data DataFrame, and plots the Close prices of the stocks along with markers for buy and sell dates. The green '^' markers represent buy dates, and the red '>' markers represent sell dates. The plot is displayed using Matplotlib library with labeled axes and a title indicating the stock ticker.

# In[20]:


import matplotlib.pyplot as plt

# Iterate through the stock_data_dict dictionary
for ticker, stock_df in stock_data_dict.items():
    buydates = stock_df[stock_df['signal'] == 'Buy'].index
    selldates = stock_df[stock_df['signal'] == 'Sell'].index

    # Plot the Close prices
    plt.figure(figsize=(12, 4))
    plt.plot(stock_df['Close'])
    plt.scatter(buydates, stock_df.loc[buydates]['Close'], marker='^', c='g', label='Buy Dates')
    plt.scatter(selldates, stock_df.loc[selldates]['Close'], marker='>', c='r', label='Sell Dates')
    plt.legend(loc=2)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'Buy and Sell Signals for {ticker}')
    plt.show()


# # Calculating Returns
# The given code calculates the total returns for each stock in the stock_data_dict dictionary. It iterates through the stock_data_dict and for each stock, it identifies the buy and sell dates based on the 'signal' column in the stock data DataFrame. It then calculates the returns by subtracting the buy prices from the corresponding sell prices. The returns are accumulated in a returns_dict dictionary with the stock ticker as the key and the total return as the value. Finally, the code prints the total returns for each stock in percentage format using string formatting.

# In[17]:


returns_dict = {}  # Dictionary to store returns for each stock

# Iterate through the stock_data_dict dictionary
for ticker, stock_df in stock_data_dict.items():
    buydates = stock_df[stock_df['signal'] == 'Buy'].index
    selldates = stock_df[stock_df['signal'] == 'Sell'].index
    buyprices = stock_df.loc[buydates]['Close']
    sellprices = stock_df.loc[selldates]['Close']

    returns = []
    for i in range(len(buyprices)):
        if i < len(sellprices):
            profit = sellprices[i] - buyprices[i]
            returns.append(profit)

    total_return = sum(returns)
    returns_dict[ticker] = total_return

# Print the returns for each stock
for ticker, total_return in returns_dict.items():
    print('Stock: {}, Total return: {:.2f}%'.format(ticker, total_return * 100)) # printing returns in percentage


# # Top performing stocks
# The given code sorts the stocks in the returns_dict dictionary by their total returns in descending order. It uses the sorted() function with a custom sorting key, which specifies that the sorting should be based on the second element of each key-value pair ( the total return) in reverse order (descending order). The sorted stocks are stored in a list called sorted_returns.
# 
# Next, the code selects the top 5 stocks from the sorted_returns list and stores them in a list called top_performing_stocks.
# 
# Finally, the code prints the ticker symbol and total return (in percentage format) for each stock in the top_performing_stocks list using string formatting, indicating the top performing stocks based on their total returns.

# In[18]:


# Sortinig  stocks by total return in descending order
sorted_returns = sorted(returns_dict.items(), key=lambda x: x[1], reverse=True)

# Selecting the top 5 stocks in terms of returns
top_n = 5

# Getting  the top performing stocks
top_performing_stocks = sorted_returns[:top_n]

# Print the top performing stocks
print('Top Performing Stocks are :')
for ticker, total_return in top_performing_stocks:
    print('Stock: {}, Total return: {:.2f}%'.format(ticker, total_return * 100))


# # Plotting Top Performing Stocks
# The code creates a line plot using Matplotlib to show the historical Close prices of the top 5 performing stocks based on their total returns. It sorts the stocks, sets up the plot with a color palette, iterates through the top performing stocks, plots their Close prices with different colors, sets labels and title, adds a legend, removes spines, sets tick label font size, and displays the plot using plt.show().

# In[19]:


import matplotlib.pyplot as plt

# Sort stocks by total return in descending order
sorted_returns = sorted(returns_dict.items(), key=lambda x: x[1], reverse=True)

# Selecting the top 5 stocks in terms of returns
top_n = 5  

# Get the top performing stocks
top_performing_stocks = sorted_returns[:top_n]

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Define color palette for lines
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plot the best performing stocks with different colors for each stock
for i, (ticker, total_return) in enumerate(top_performing_stocks):
    stock_data = stock_data_dict[ticker]
    ax.plot(stock_data['Close'], label=ticker, color=colors[i % len(colors)], linewidth=2)

# Set x-axis label
ax.set_xlabel('Date', fontsize=14)

# Set y-axis label
ax.set_ylabel('Close Price', fontsize=14)

# Set title
ax.set_title('Best Performing Stocks', fontsize=16)

# Add legend
ax.legend(loc='upper left', fontsize=12)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set tick label font size
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# Show the plot
plt.show()

